//this is the section of the program that converts data to or from a format suitable for transmission to or reciept from other programs
use crate::internal;
use serde::{Deserialize, Serialize};
use serde_json_any_key::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct NodeMut {
    pub visibility: bool,
    pub flavor: usize, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub units: Vec<Unit>,
    pub factoryinstancelist: Vec<FactoryInstance>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub allegiance: usize, //faction that currently holds the node
    pub efficiency: f32, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub balance_stockpiles: bool,
    pub balance_hangars: bool,
    pub check_for_battles: bool,
    pub stockpiles_balanced: bool,
    pub hangars_balanced: bool,
}

impl NodeMut {
    fn desiccate(self_entity: &internal::NodeMut) -> NodeMut {
        NodeMut {
            visibility: self_entity.visibility,
            flavor: self_entity.flavor.id,
            units: self_entity
                .units
                .iter()
                .map(|x| Unit::desiccate(x))
                .collect(),
            factoryinstancelist: self_entity
                .factoryinstancelist
                .iter()
                .map(|x| FactoryInstance::desiccate(x))
                .collect(),
            shipyardinstancelist: self_entity
                .shipyardinstancelist
                .iter()
                .map(|x| ShipyardInstance::desiccate(x))
                .collect(),
            allegiance: self_entity.allegiance.id,
            efficiency: self_entity.efficiency,
            balance_stockpiles: self_entity.balance_stockpiles,
            balance_hangars: self_entity.balance_hangars,
            check_for_battles: self_entity.check_for_battles,
            stockpiles_balanced: self_entity.stockpiles_balanced,
            hangars_balanced: self_entity.hangars_balanced,
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<internal::NodeFlavor>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        factoryclassesroot: &Vec<Arc<internal::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<internal::ShipyardClass>>,
        shipclassesroot: &Vec<Arc<internal::ShipClass>>,
    ) -> internal::NodeMut {
        internal::NodeMut {
            visibility: self.visibility,
            flavor: nodeflavorsroot[self.flavor].clone(),
            units: Vec::new(),
            factoryinstancelist: self
                .factoryinstancelist
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &factoryclassesroot))
                .collect(),
            shipyardinstancelist: self
                .shipyardinstancelist
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &shipyardclassesroot, &shipclassesroot))
                .collect(),
            allegiance: factionsroot[self.allegiance].clone(),
            efficiency: self.efficiency,
            balance_stockpiles: self.balance_stockpiles,
            balance_hangars: self.balance_hangars,
            check_for_battles: self.check_for_battles,
            stockpiles_balanced: self.stockpiles_balanced,
            hangars_balanced: self.hangars_balanced,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub visiblename: String, //location name as shown to player
    pub position: [i64; 3], //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    pub description: String,
    pub environment: String, //name of the FRED environment to use for missions set in this node
    pub bitmap: Option<(String, f32)>, //name of the bitmap used to depict this node in other nodes' FRED environments, and a size factor
    pub mutables: NodeMut,
}

impl Node {
    fn desiccate(self_entity: &internal::Node) -> Node {
        Node {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            position: self_entity.position,
            description: self_entity.description.clone(),
            environment: self_entity.environment.clone(),
            bitmap: self_entity.bitmap.clone(),
            mutables: NodeMut::desiccate(&self_entity.mutables.read().unwrap()),
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<internal::NodeFlavor>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        factoryclassesroot: &Vec<Arc<internal::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<internal::ShipyardClass>>,
        shipclassesroot: &Vec<Arc<internal::ShipClass>>,
    ) -> internal::Node {
        internal::Node {
            id: self.id,
            visiblename: self.visiblename.clone(),
            position: self.position,
            description: self.description.clone(),
            environment: self.environment.clone(),
            bitmap: self.bitmap.clone(),
            mutables: RwLock::new(self.mutables.rehydrate(
                &nodeflavorsroot,
                &factionsroot,
                &resourcesroot,
                &factoryclassesroot,
                &shipyardclassesroot,
                &shipclassesroot,
            )),
        }
    }
    fn add_units(
        node: &Arc<internal::Node>,
        connectionnodes: &Vec<Node>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) {
        node.mutables.write().unwrap().units = connectionnodes[node.id]
            .mutables
            .units
            .iter()
            .map(|x| x.rehydrate(&shipinstancesroot, &squadroninstancesroot))
            .collect();
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct System {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub nodes: Vec<usize>,
}

impl System {
    fn desiccate(self_entity: &internal::System) -> System {
        System {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility.clone(),
            nodes: self_entity.nodes.iter().map(|x| x.id).collect(),
        }
    }
    fn rehydrate(&self, nodesroot: &Vec<Arc<internal::Node>>) -> internal::System {
        internal::System {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            visibility: self.visibility.clone(),
            nodes: self.nodes.iter().map(|x| nodesroot[*x].clone()).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edges {
    hyperlinks: HashSet<(usize, usize, usize)>, //list of links between nodes
    neighbormap: HashMap<usize, Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnipotentStockpile {
    pub visibility: bool,
    pub resourcetype: usize,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl UnipotentStockpile {
    fn desiccate(self_entity: &internal::UnipotentStockpile) -> UnipotentStockpile {
        UnipotentStockpile {
            visibility: self_entity.visibility,
            resourcetype: self_entity.resourcetype.id,
            contents: self_entity.contents,
            rate: self_entity.rate,
            target: self_entity.target,
            capacity: self_entity.capacity,
            propagates: self_entity.propagates,
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<internal::Resource>>,
    ) -> internal::UnipotentStockpile {
        internal::UnipotentStockpile {
            visibility: self.visibility,
            resourcetype: resourcesroot[self.resourcetype].clone(),
            contents: self.contents,
            rate: self.rate,
            target: self.target,
            capacity: self.capacity,
            propagates: self.propagates,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PluripotentStockpile {
    pub visibility: bool,
    pub contents: HashMap<usize, u64>,
    pub allowed: Option<Vec<usize>>,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl PluripotentStockpile {
    fn desiccate(self_entity: &internal::PluripotentStockpile) -> PluripotentStockpile {
        PluripotentStockpile {
            visibility: self_entity.visibility,
            contents: self_entity
                .contents
                .iter()
                .map(|(resource, count)| (resource.id, *count))
                .collect(),
            allowed: self_entity
                .allowed
                .clone()
                .map(|vec| vec.iter().map(|x| x.id).collect()),
            target: self_entity.target,
            capacity: self_entity.capacity,
            propagates: self_entity.propagates,
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<internal::Resource>>,
    ) -> internal::PluripotentStockpile {
        internal::PluripotentStockpile {
            visibility: self.visibility,
            contents: self
                .contents
                .iter()
                .map(|(resource, count)| (resourcesroot[*resource].clone(), *count))
                .collect(),
            allowed: self
                .allowed
                .clone()
                .map(|vec| vec.iter().map(|x| resourcesroot[*x].clone()).collect()),
            target: self.target,
            capacity: self.capacity,
            propagates: self.propagates,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SharedStockpile {
    pub resourcetype: usize,
    pub contents: u64,
    pub rate: u64,
    pub capacity: u64,
}

impl SharedStockpile {
    fn desiccate(self_entity: &internal::SharedStockpile) -> SharedStockpile {
        SharedStockpile {
            resourcetype: self_entity.resourcetype.id,
            contents: self_entity.contents.load(atomic::Ordering::Relaxed),
            rate: self_entity.rate,
            capacity: self_entity.capacity,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<internal::Resource>>) -> internal::SharedStockpile {
        internal::SharedStockpile {
            resourcetype: resourcesroot[self.resourcetype].clone(),
            contents: Arc::new(AtomicU64::new(self.contents)),
            rate: self.rate,
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HangarInstanceMut {
    pub visibility: bool,
    pub contents: Vec<Unit>,
}

impl HangarInstanceMut {
    fn desiccate(self_entity: &internal::HangarInstanceMut) -> HangarInstanceMut {
        HangarInstanceMut {
            visibility: self_entity.visibility,
            contents: self_entity
                .contents
                .iter()
                .map(|x| Unit::desiccate(x))
                .collect(),
        }
    }
    fn rehydrate(
        &self,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> internal::HangarInstanceMut {
        internal::HangarInstanceMut {
            visibility: self.visibility,
            contents: self
                .contents
                .iter()
                .map(|x| x.rehydrate(&shipinstancesroot, &squadroninstancesroot))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HangarInstance {
    pub id: u64,
    pub class: usize,
    pub mother: u64,
    pub mutables: HangarInstanceMut,
}

impl HangarInstance {
    pub fn desiccate(self_entity: &internal::HangarInstance) -> HangarInstance {
        HangarInstance {
            id: self_entity.id,
            class: self_entity.class.id,
            mother: self_entity.mother.id,
            mutables: HangarInstanceMut::desiccate(&self_entity.mutables.read().unwrap()),
        }
    }
    pub fn rehydrate(
        &self,
        hangarclassesroot: &Vec<Arc<internal::HangarClass>>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> internal::HangarInstance {
        internal::HangarInstance {
            id: self.id,
            class: hangarclassesroot[self.class].clone(),
            mother: shipinstancesroot[self.mother as usize].clone(),
            mutables: RwLock::new(
                self.mutables
                    .rehydrate(&shipinstancesroot, &squadroninstancesroot),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub basehealth: Option<u64>,
    pub toughnessscalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<usize>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<usize>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl EngineClass {
    pub fn desiccate(self_entity: &internal::EngineClass) -> EngineClass {
        EngineClass {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            basehealth: self_entity.basehealth,
            toughnessscalar: self_entity.toughnessscalar.clone(),
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            forbidden_nodeflavors: self_entity
                .forbidden_nodeflavors
                .iter()
                .map(|x| x.id)
                .collect(),
            forbidden_edgeflavors: self_entity
                .forbidden_edgeflavors
                .iter()
                .map(|x| x.id)
                .collect(),
            speed: self_entity.speed,
            cooldown: self_entity.cooldown,
        }
    }
    pub fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<internal::NodeFlavor>>,
        edgeflavorsroot: &Vec<Arc<internal::EdgeFlavor>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
    ) -> internal::EngineClass {
        internal::EngineClass {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
            basehealth: self.basehealth,
            toughnessscalar: self.toughnessscalar.clone(),
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            forbidden_nodeflavors: self
                .forbidden_nodeflavors
                .iter()
                .map(|x| nodeflavorsroot[*x].clone())
                .collect(),
            forbidden_edgeflavors: self
                .forbidden_edgeflavors
                .iter()
                .map(|x| edgeflavorsroot[*x].clone())
                .collect(),
            speed: self.speed,
            cooldown: self.cooldown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineInstance {
    pub engineclass: usize,
    pub visibility: bool,
    pub basehealth: Option<u64>,
    pub health: Option<u64>,
    pub toughnessscalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<usize>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<usize>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
    pub last_move_turn: u64,
}

impl EngineInstance {
    fn desiccate(self_entity: &internal::EngineInstance) -> EngineInstance {
        EngineInstance {
            engineclass: self_entity.engineclass.id,
            visibility: self_entity.visibility,
            basehealth: self_entity.basehealth,
            health: self_entity.health,
            toughnessscalar: self_entity.toughnessscalar,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            forbidden_nodeflavors: self_entity
                .forbidden_nodeflavors
                .iter()
                .map(|x| x.id)
                .collect(),
            forbidden_edgeflavors: self_entity
                .forbidden_edgeflavors
                .iter()
                .map(|x| x.id)
                .collect(),
            speed: self_entity.speed,
            cooldown: self_entity.cooldown,
            last_move_turn: self_entity.last_move_turn,
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<internal::NodeFlavor>>,
        edgeflavorsroot: &Vec<Arc<internal::EdgeFlavor>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        engineclassesroot: &Vec<Arc<internal::EngineClass>>,
    ) -> internal::EngineInstance {
        internal::EngineInstance {
            engineclass: engineclassesroot[self.engineclass].clone(),
            visibility: self.visibility,
            basehealth: self.basehealth,
            health: self.health,
            toughnessscalar: self.toughnessscalar,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            forbidden_nodeflavors: self
                .forbidden_nodeflavors
                .iter()
                .map(|x| nodeflavorsroot[*x].clone())
                .collect(),
            forbidden_edgeflavors: self
                .forbidden_edgeflavors
                .iter()
                .map(|x| edgeflavorsroot[*x].clone())
                .collect(),
            speed: self.speed,
            cooldown: self.cooldown,
            last_move_turn: self.last_move_turn,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairerClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerClass {
    fn desiccate(self_entity: &internal::RepairerClass) -> RepairerClass {
        RepairerClass {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            repair_points: self_entity.repair_points,
            repair_factor: self_entity.repair_factor.clone(),
            engine_repair_points: self_entity.engine_repair_points,
            engine_repair_factor: self_entity.engine_repair_factor.clone(),
            per_engagement: self_entity.per_engagement,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<internal::Resource>>) -> internal::RepairerClass {
        internal::RepairerClass {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            repair_points: self.repair_points,
            repair_factor: self.repair_factor.clone(),
            engine_repair_points: self.engine_repair_points,
            engine_repair_factor: self.engine_repair_factor.clone(),
            per_engagement: self.per_engagement,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairerInstance {
    pub repairerclass: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerInstance {
    fn desiccate(self_entity: &internal::RepairerInstance) -> RepairerInstance {
        RepairerInstance {
            repairerclass: self_entity.repairerclass.id,
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            repair_points: self_entity.repair_points,
            repair_factor: self_entity.repair_factor.clone(),
            engine_repair_points: self_entity.engine_repair_points,
            engine_repair_factor: self_entity.engine_repair_factor.clone(),
            per_engagement: self_entity.per_engagement,
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        repairerclassesroot: &Vec<Arc<internal::RepairerClass>>,
    ) -> internal::RepairerInstance {
        internal::RepairerInstance {
            repairerclass: repairerclassesroot[self.repairerclass].clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            repair_points: self.repair_points,
            repair_factor: self.repair_factor.clone(),
            engine_repair_points: self.engine_repair_points,
            engine_repair_factor: self.engine_repair_factor.clone(),
            per_engagement: self.per_engagement,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoryClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    fn desiccate(self_entity: &internal::FactoryClass) -> FactoryClass {
        FactoryClass {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            outputs: self_entity
                .outputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<internal::Resource>>) -> internal::FactoryClass {
        internal::FactoryClass {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoryInstance {
    //this is an actual factory, derived from a factory class
    pub factoryclass: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryInstance {
    fn desiccate(self_entity: &internal::FactoryInstance) -> FactoryInstance {
        FactoryInstance {
            factoryclass: self_entity.factoryclass.id,
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            outputs: self_entity
                .outputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        factoryclassesroot: &Vec<Arc<internal::FactoryClass>>,
    ) -> internal::FactoryInstance {
        internal::FactoryInstance {
            factoryclass: factoryclassesroot[self.factoryclass].clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipyardClass {
    pub id: usize,
    pub visiblename: Option<String>,
    pub description: Option<String>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<internal::ShipClassID, u64>,
    pub constructrate: u64,
    pub efficiency: f32,
}

impl ShipyardClass {
    fn desiccate(self_entity: &internal::ShipyardClass) -> ShipyardClass {
        ShipyardClass {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            outputs: self_entity.outputs.clone(),
            constructrate: self_entity.constructrate,
            efficiency: self_entity.efficiency,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<internal::Resource>>) -> internal::ShipyardClass {
        internal::ShipyardClass {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            outputs: self.outputs.clone(),
            constructrate: self.constructrate,
            efficiency: self.efficiency,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipyardInstance {
    pub shipyardclass: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<usize, u64>,
    pub constructpoints: u64,
    pub constructrate: u64,
    pub efficiency: f32,
}

impl ShipyardInstance {
    fn desiccate(self_entity: &internal::ShipyardInstance) -> ShipyardInstance {
        ShipyardInstance {
            shipyardclass: self_entity.shipyardclass.id,
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            outputs: self_entity
                .outputs
                .iter()
                .map(|(shipclass, count)| (shipclass.id, *count))
                .collect(),
            constructpoints: self_entity.constructpoints,
            constructrate: self_entity.constructrate,
            efficiency: self_entity.efficiency,
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        shipyardclassesroot: &Vec<Arc<internal::ShipyardClass>>,
        shipclassesroot: &Vec<Arc<internal::ShipClass>>,
    ) -> internal::ShipyardInstance {
        internal::ShipyardInstance {
            shipyardclass: shipyardclassesroot[self.shipyardclass].clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|(shipclass, count)| (shipclassesroot[*shipclass].clone(), *count))
                .collect(),
            constructpoints: self.constructpoints,
            constructrate: self.constructrate,
            efficiency: self.efficiency,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipAI {
    pub id: usize,
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<internal::UnitClassID, f32>,
    pub resource_attract: HashMap<usize, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
    pub friendly_supply_attract: f32,
    pub hostile_supply_attract: f32,
    pub allegiance_demand_attract: f32,
    pub enemy_demand_attract: f32,
}

impl ShipAI {
    fn desiccate(self_entity: &internal::ShipAI) -> ShipAI {
        ShipAI {
            id: self_entity.id,
            ship_attract_specific: self_entity.ship_attract_specific,
            ship_attract_generic: self_entity.ship_attract_generic,
            ship_cargo_attract: self_entity.ship_cargo_attract.clone(),
            resource_attract: self_entity
                .resource_attract
                .iter()
                .map(|(resource, scalar)| (resource.id, *scalar))
                .collect(),
            friendly_supply_attract: self_entity.friendly_supply_attract,
            hostile_supply_attract: self_entity.hostile_supply_attract,
            allegiance_demand_attract: self_entity.allegiance_demand_attract,
            enemy_demand_attract: self_entity.enemy_demand_attract,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<internal::Resource>>) -> internal::ShipAI {
        internal::ShipAI {
            id: self.id,
            ship_attract_specific: self.ship_attract_specific,
            ship_attract_generic: self.ship_attract_generic,
            ship_cargo_attract: self.ship_cargo_attract.clone(),
            resource_attract: self
                .resource_attract
                .iter()
                .map(|(resource, scalar)| (resourcesroot[*resource].clone(), *scalar))
                .collect(),
            friendly_supply_attract: self.friendly_supply_attract,
            hostile_supply_attract: self.hostile_supply_attract,
            allegiance_demand_attract: self.allegiance_demand_attract,
            enemy_demand_attract: self.enemy_demand_attract,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum UnitLocation {
    Node(usize),
    Squadron(u64),
    Hangar(u64),
}

impl UnitLocation {
    fn desiccate(self_entity: &internal::UnitLocation) -> UnitLocation {
        match self_entity {
            internal::UnitLocation::Node(n) => UnitLocation::Node(n.id),
            internal::UnitLocation::Squadron(s) => UnitLocation::Squadron(s.id),
            internal::UnitLocation::Hangar(h) => UnitLocation::Hangar(h.id),
        }
    }
    fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
        hangarslist: &Vec<Arc<internal::HangarInstance>>,
    ) -> internal::UnitLocation {
        match self {
            UnitLocation::Node(n) => internal::UnitLocation::Node(nodesroot[*n].clone()),
            UnitLocation::Squadron(s) => {
                internal::UnitLocation::Squadron(squadroninstancesroot[*s as usize].clone())
            }
            UnitLocation::Hangar(h) => {
                internal::UnitLocation::Hangar(hangarslist[*h as usize].clone())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub shipflavor: usize,
    pub basehull: u64,     //how many hull hitpoints this ship has by default
    pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub propagates: bool,
    pub hangarvol: u64,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub defaultweapons: Option<HashMap<usize, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<usize>,
    pub engines: Vec<usize>,
    pub repairers: Vec<usize>,
    pub factoryclasslist: Vec<usize>,
    pub shipyardclasslist: Vec<usize>,
    pub aiclass: usize,
    pub navthreshold: f32, //the value of an adjacent node must exceed (the value of the current node times navthreshold) in order for the ship to decide to move
    pub processordemandnavscalar: f32, //multiplier for demand generated by the ship's engines, repairers, factories, and shipyards, to modify it relative to that generated by stockpiles
    pub deploys_self: bool,            //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments; value is number of moves a daughter must be able to make to be deployed
    pub defectchance: HashMap<usize, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub toughnessscalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub battleescapescalar: f32, //is added to toughnessscalar in battles where this ship is on the losing side, trying to escape
    pub defectescapescalar: f32, //influences how likely it is that a ship of this class will, if it defects, escape to an enemy-held node with no engagement taking place
    pub interdictionscalar: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this shipclass to be
}

impl ShipClass {
    fn desiccate(self_entity: &internal::ShipClass) -> ShipClass {
        ShipClass {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            shipflavor: self_entity.shipflavor.id,
            basehull: self_entity.basehull,
            basestrength: self_entity.basestrength,
            visibility: self_entity.visibility,
            propagates: self_entity.propagates,
            hangarvol: self_entity.hangarvol,
            stockpiles: self_entity
                .stockpiles
                .iter()
                .map(|x| PluripotentStockpile::desiccate(x))
                .collect(),
            defaultweapons: self_entity.defaultweapons.clone().map(|x| {
                x.iter()
                    .map(|(resource, count)| (resource.id, *count))
                    .collect()
            }),
            hangars: self_entity.hangars.iter().map(|x| x.id).collect(),
            engines: self_entity.engines.iter().map(|x| x.id).collect(),
            repairers: self_entity.repairers.iter().map(|x| x.id).collect(),
            factoryclasslist: self_entity.factoryclasslist.iter().map(|x| x.id).collect(),
            shipyardclasslist: self_entity.shipyardclasslist.iter().map(|x| x.id).collect(),
            aiclass: self_entity.aiclass.id,
            navthreshold: self_entity.navthreshold.clone(),
            processordemandnavscalar: self_entity.processordemandnavscalar.clone(),
            deploys_self: self_entity.deploys_self,
            deploys_daughters: self_entity.deploys_daughters,
            defectchance: self_entity
                .defectchance
                .iter()
                .map(|(faction, scalars)| (faction.id, *scalars))
                .collect(),
            toughnessscalar: self_entity.toughnessscalar.clone(),
            battleescapescalar: self_entity.battleescapescalar.clone(),
            defectescapescalar: self_entity.defectescapescalar.clone(),
            interdictionscalar: self_entity.interdictionscalar.clone(),
            value_mult: self_entity.value_mult.clone(),
        }
    }
    fn rehydrate(
        &self,
        factionsroot: &Vec<Arc<internal::Faction>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        hangarclassesroot: &Vec<Arc<internal::HangarClass>>,
        engineclassesroot: &Vec<Arc<internal::EngineClass>>,
        repairerclassesroot: &Vec<Arc<internal::RepairerClass>>,
        factoryclassesroot: &Vec<Arc<internal::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<internal::ShipyardClass>>,
        shipaisroot: &Vec<Arc<internal::ShipAI>>,
        shipflavorsroot: &Vec<Arc<internal::ShipFlavor>>,
    ) -> internal::ShipClass {
        internal::ShipClass {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            shipflavor: shipflavorsroot[self.shipflavor].clone(),
            basehull: self.basehull,
            basestrength: self.basestrength,
            visibility: self.visibility,
            propagates: self.propagates,
            hangarvol: self.hangarvol,
            stockpiles: self
                .stockpiles
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            defaultweapons: self.defaultweapons.clone().map(|x| {
                x.iter()
                    .map(|(resource, count)| (resourcesroot[*resource].clone(), *count))
                    .collect()
            }),
            hangars: self
                .hangars
                .iter()
                .map(|x| hangarclassesroot[*x].clone())
                .collect(),
            engines: self
                .engines
                .iter()
                .map(|x| engineclassesroot[*x].clone())
                .collect(),
            repairers: self
                .repairers
                .iter()
                .map(|x| repairerclassesroot[*x].clone())
                .collect(),
            factoryclasslist: self
                .factoryclasslist
                .iter()
                .map(|x| factoryclassesroot[*x].clone())
                .collect(),
            shipyardclasslist: self
                .shipyardclasslist
                .iter()
                .map(|x| shipyardclassesroot[*x].clone())
                .collect(),
            aiclass: shipaisroot[self.aiclass].clone(),
            navthreshold: self.navthreshold.clone(),
            processordemandnavscalar: self.processordemandnavscalar.clone(),
            deploys_self: self.deploys_self,
            deploys_daughters: self.deploys_daughters,
            defectchance: self
                .defectchance
                .iter()
                .map(|(faction, scalars)| (factionsroot[*faction].clone(), *scalars))
                .collect(),
            toughnessscalar: self.toughnessscalar.clone(),
            battleescapescalar: self.battleescapescalar.clone(),
            defectescapescalar: self.defectescapescalar.clone(),
            interdictionscalar: self.interdictionscalar.clone(),
            value_mult: self.value_mult.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipInstanceMut {
    pub hull: u64, //how many hitpoints the ship has
    pub visibility: bool,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<HangarInstance>,
    pub engines: Vec<EngineInstance>,
    pub movement_left: u64, //starts at eighteen quintillion each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<RepairerInstance>,
    pub factoryinstancelist: Vec<FactoryInstance>,
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub location: UnitLocation, //where the ship is -- a node if it's unaffiliated, a squadron if it's in one
    pub allegiance: usize,      //which faction this ship belongs to
    pub objectives: Vec<Objective>,
    pub aiclass: usize,
}

impl ShipInstanceMut {
    fn desiccate(self_entity: &internal::ShipInstanceMut) -> ShipInstanceMut {
        ShipInstanceMut {
            hull: self_entity.hull,
            visibility: self_entity.visibility,
            stockpiles: self_entity
                .stockpiles
                .iter()
                .map(|x| PluripotentStockpile::desiccate(x))
                .collect(),
            efficiency: self_entity.efficiency.clone(),
            hangars: self_entity
                .hangars
                .iter()
                .map(|x| HangarInstance::desiccate(x))
                .collect(),
            engines: self_entity
                .engines
                .iter()
                .map(|x| EngineInstance::desiccate(x))
                .collect(),
            movement_left: self_entity.movement_left,
            repairers: self_entity
                .repairers
                .iter()
                .map(|x| RepairerInstance::desiccate(x))
                .collect(),
            factoryinstancelist: self_entity
                .factoryinstancelist
                .iter()
                .map(|x| FactoryInstance::desiccate(x))
                .collect(),
            shipyardinstancelist: self_entity
                .shipyardinstancelist
                .iter()
                .map(|x| ShipyardInstance::desiccate(x))
                .collect(),
            location: UnitLocation::desiccate(&self_entity.location),
            allegiance: self_entity.allegiance.id,
            objectives: self_entity
                .objectives
                .iter()
                .map(|x| Objective::desiccate(x))
                .collect(),
            aiclass: self_entity.aiclass.id,
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<internal::NodeFlavor>>,
        nodesroot: &Vec<Arc<internal::Node>>,
        edgeflavorsroot: &Vec<Arc<internal::EdgeFlavor>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        engineclassesroot: &Vec<Arc<internal::EngineClass>>,
        repairerclassesroot: &Vec<Arc<internal::RepairerClass>>,
        factoryclassesroot: &Vec<Arc<internal::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<internal::ShipyardClass>>,
        shipaisroot: &Vec<Arc<internal::ShipAI>>,
        shipclassesroot: &Vec<Arc<internal::ShipClass>>,
    ) -> internal::ShipInstanceMut {
        internal::ShipInstanceMut {
            hull: self.hull,
            visibility: self.visibility,
            stockpiles: self
                .stockpiles
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            efficiency: self.efficiency.clone(),
            hangars: Vec::new(),
            engines: self
                .engines
                .iter()
                .map(|x| {
                    x.rehydrate(
                        &nodeflavorsroot,
                        &edgeflavorsroot,
                        &resourcesroot,
                        &engineclassesroot,
                    )
                })
                .collect(),
            movement_left: self.movement_left,
            repairers: self
                .repairers
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &repairerclassesroot))
                .collect(),
            factoryinstancelist: self
                .factoryinstancelist
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &factoryclassesroot))
                .collect(),
            shipyardinstancelist: self
                .shipyardinstancelist
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &shipyardclassesroot, &shipclassesroot))
                .collect(),
            location: internal::UnitLocation::Node(nodesroot[0].clone()),
            allegiance: factionsroot[self.allegiance].clone(),
            objectives: Vec::new(),
            aiclass: shipaisroot[self.aiclass].clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: usize, //which class of ship this is
    pub mutables: ShipInstanceMut,
}

impl ShipInstance {
    fn desiccate(self_entity: &internal::ShipInstance) -> ShipInstance {
        ShipInstance {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            class: self_entity.class.id,
            mutables: ShipInstanceMut::desiccate(&self_entity.mutables.read().unwrap()),
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<internal::NodeFlavor>>,
        nodesroot: &Vec<Arc<internal::Node>>,
        edgeflavorsroot: &Vec<Arc<internal::EdgeFlavor>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
        resourcesroot: &Vec<Arc<internal::Resource>>,
        engineclassesroot: &Vec<Arc<internal::EngineClass>>,
        repairerclassesroot: &Vec<Arc<internal::RepairerClass>>,
        factoryclassesroot: &Vec<Arc<internal::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<internal::ShipyardClass>>,
        shipaisroot: &Vec<Arc<internal::ShipAI>>,
        shipclassesroot: &Vec<Arc<internal::ShipClass>>,
    ) -> internal::ShipInstance {
        internal::ShipInstance {
            id: self.id,
            visiblename: self.visiblename.clone(),
            class: shipclassesroot[self.class].clone(),
            mutables: RwLock::new(self.mutables.rehydrate(
                &nodeflavorsroot,
                &nodesroot,
                &edgeflavorsroot,
                &factionsroot,
                &resourcesroot,
                &engineclassesroot,
                &repairerclassesroot,
                &factoryclassesroot,
                &shipyardclassesroot,
                &shipaisroot,
                &shipclassesroot,
            )),
        }
    }
    fn add_hangars_and_objectives(
        ship: &Arc<internal::ShipInstance>,
        connectionships: &Vec<ShipInstance>,
        nodesroot: &Vec<Arc<internal::Node>>,
        systemsroot: &Vec<Arc<internal::System>>,
        hangarclassesroot: &Vec<Arc<internal::HangarClass>>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> Vec<Arc<internal::HangarInstance>> {
        let internal_hangars: Vec<Arc<internal::HangarInstance>> = connectionships
            [ship.id as usize]
            .mutables
            .hangars
            .iter()
            .map(|x| {
                Arc::new(x.rehydrate(
                    &hangarclassesroot,
                    &shipinstancesroot,
                    &squadroninstancesroot,
                ))
            })
            .collect();
        ship.mutables.write().unwrap().hangars = internal_hangars.clone();
        ship.mutables.write().unwrap().objectives = connectionships[ship.id as usize]
            .mutables
            .objectives
            .iter()
            .map(|x| {
                x.rehydrate(
                    &nodesroot,
                    &systemsroot,
                    &shipinstancesroot,
                    &squadroninstancesroot,
                )
            })
            .collect();
        internal_hangars
    }
    fn set_location(
        ship: &Arc<internal::ShipInstance>,
        connectionships: &Vec<ShipInstance>,
        nodesroot: &Vec<Arc<internal::Node>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
        hangarslist: &Vec<Arc<internal::HangarInstance>>,
    ) {
        ship.mutables.write().unwrap().location = connectionships[ship.id as usize]
            .mutables
            .location
            .rehydrate(&nodesroot, &squadroninstancesroot, &hangarslist);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub squadronflavor: usize,
    pub visibility: bool,
    pub propagates: bool,
    pub strengthmod: (f32, u64),
    pub squadronconfig: HashMap<internal::UnitClassID, u64>,
    pub non_ideal_supply_scalar: f32, //multiplier used for demand generated by non-ideal ships under the target limit; should be below one
    pub target: u64,
    pub navthreshold: f32, //the value of an adjacent node must exceed (the value of the current node times navthreshold) in order for the ship to decide to move
    pub navquorum: f32,
    pub disbandthreshold: f32,
    pub deploys_self: bool, //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments
    pub defectchance: HashMap<usize, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub defectescapescalar: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this squadronclass to be
}

impl SquadronClass {
    fn desiccate(self_entity: &internal::SquadronClass) -> SquadronClass {
        SquadronClass {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            description: self_entity.description.clone(),
            squadronflavor: self_entity.squadronflavor.id,
            visibility: self_entity.visibility,
            propagates: self_entity.propagates,
            strengthmod: self_entity.strengthmod.clone(),
            squadronconfig: self_entity.squadronconfig.clone(),
            non_ideal_supply_scalar: self_entity.non_ideal_supply_scalar.clone(),
            target: self_entity.target,
            navthreshold: self_entity.navthreshold.clone(),
            navquorum: self_entity.navquorum.clone(),
            disbandthreshold: self_entity.disbandthreshold.clone(),
            deploys_self: self_entity.deploys_self,
            deploys_daughters: self_entity.deploys_daughters,
            defectchance: self_entity
                .defectchance
                .iter()
                .map(|(faction, scalars)| (faction.id, *scalars))
                .collect(),
            defectescapescalar: self_entity.defectescapescalar.clone(),
            value_mult: self_entity.value_mult.clone(),
        }
    }
    fn rehydrate(
        &self,
        factionsroot: &Vec<Arc<internal::Faction>>,
        squadronflavorsroot: &Vec<Arc<internal::SquadronFlavor>>,
    ) -> internal::SquadronClass {
        internal::SquadronClass {
            id: self.id,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            squadronflavor: squadronflavorsroot[self.squadronflavor].clone(),
            visibility: self.visibility,
            propagates: self.propagates,
            strengthmod: self.strengthmod.clone(),
            squadronconfig: self.squadronconfig.clone(),
            non_ideal_supply_scalar: self.non_ideal_supply_scalar.clone(),
            target: self.target,
            navthreshold: self.navthreshold.clone(),
            navquorum: self.navquorum.clone(),
            disbandthreshold: self.disbandthreshold.clone(),
            deploys_self: self.deploys_self,
            deploys_daughters: self.deploys_daughters,
            defectchance: self
                .defectchance
                .iter()
                .map(|(faction, scalars)| (factionsroot[*faction].clone(), *scalars))
                .collect(),
            defectescapescalar: self.defectescapescalar.clone(),
            value_mult: self.value_mult.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronInstanceMut {
    pub visibility: bool,
    pub location: UnitLocation,
    pub daughters: Vec<Unit>,
    pub allegiance: usize,
    pub objectives: Vec<Objective>,
    pub ghost: bool,
}

impl SquadronInstanceMut {
    fn desiccate(self_entity: &internal::SquadronInstanceMut) -> SquadronInstanceMut {
        SquadronInstanceMut {
            visibility: self_entity.visibility,
            location: UnitLocation::desiccate(&self_entity.location),
            daughters: self_entity
                .daughters
                .iter()
                .map(|x| Unit::desiccate(x))
                .collect(),
            allegiance: self_entity.allegiance.id,
            objectives: self_entity
                .objectives
                .iter()
                .map(|x| Objective::desiccate(x))
                .collect(),
            ghost: self_entity.ghost,
        }
    }
    fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
    ) -> internal::SquadronInstanceMut {
        internal::SquadronInstanceMut {
            visibility: self.visibility,
            location: internal::UnitLocation::Node(nodesroot[0].clone()),
            daughters: Vec::new(),
            allegiance: factionsroot[self.allegiance].clone(),
            objectives: Vec::new(),
            ghost: self.ghost,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: usize,
    pub idealstrength: u64,
    pub mutables: SquadronInstanceMut,
}

impl SquadronInstance {
    fn desiccate(self_entity: &internal::SquadronInstance) -> SquadronInstance {
        SquadronInstance {
            id: self_entity.id,
            visiblename: self_entity.visiblename.clone(),
            class: self_entity.class.id,
            idealstrength: self_entity.idealstrength,
            mutables: SquadronInstanceMut::desiccate(&self_entity.mutables.read().unwrap()),
        }
    }
    fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
        squadronclassesroot: &Vec<Arc<internal::SquadronClass>>,
    ) -> internal::SquadronInstance {
        internal::SquadronInstance {
            id: self.id,
            visiblename: self.visiblename.clone(),
            class: squadronclassesroot[self.class].clone(),
            idealstrength: self.idealstrength,
            mutables: RwLock::new(self.mutables.rehydrate(&nodesroot, &factionsroot)),
        }
    }
    fn add_daughters_and_objectives_set_location(
        squadron: &Arc<internal::SquadronInstance>,
        connectionsquadrons: &Vec<SquadronInstance>,
        nodesroot: &Vec<Arc<internal::Node>>,
        systemsroot: &Vec<Arc<internal::System>>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
        hangarslist: &Vec<Arc<internal::HangarInstance>>,
    ) {
        squadron.mutables.write().unwrap().daughters = connectionsquadrons[squadron.id as usize]
            .mutables
            .daughters
            .iter()
            .map(|x| x.rehydrate(&shipinstancesroot, &squadroninstancesroot))
            .collect();
        squadron.mutables.write().unwrap().objectives = connectionsquadrons[squadron.id as usize]
            .mutables
            .objectives
            .iter()
            .map(|x| {
                x.rehydrate(
                    &nodesroot,
                    &systemsroot,
                    &shipinstancesroot,
                    &squadroninstancesroot,
                )
            })
            .collect();
        squadron.mutables.write().unwrap().location = connectionsquadrons[squadron.id as usize]
            .mutables
            .location
            .rehydrate(&nodesroot, &squadroninstancesroot, &hangarslist);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum UnitClass {
    ShipClass(usize),
    SquadronClass(usize),
}

impl UnitClass {
    fn desiccate(self_entity: &internal::UnitClass) -> UnitClass {
        match self_entity {
            internal::UnitClass::ShipClass(shc) => UnitClass::ShipClass(shc.id),
            internal::UnitClass::SquadronClass(sqc) => UnitClass::SquadronClass(sqc.id),
        }
    }
    fn rehydrate(
        &self,
        shipclassesroot: &Vec<Arc<internal::ShipClass>>,
        squadronclassesroot: &Vec<Arc<internal::SquadronClass>>,
    ) -> internal::UnitClass {
        match self {
            UnitClass::ShipClass(shc) => {
                internal::UnitClass::ShipClass(shipclassesroot[*shc].clone())
            }
            UnitClass::SquadronClass(sqc) => {
                internal::UnitClass::SquadronClass(squadronclassesroot[*sqc].clone())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum Unit {
    Ship(u64),
    Squadron(u64),
}

impl Unit {
    fn desiccate(self_entity: &internal::Unit) -> Unit {
        match self_entity {
            internal::Unit::Ship(sh) => Unit::Ship(sh.id),
            internal::Unit::Squadron(sq) => Unit::Squadron(sq.id),
        }
    }
    fn rehydrate(
        &self,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> internal::Unit {
        match self {
            Unit::Ship(sh) => internal::Unit::Ship(shipinstancesroot[*sh as usize].clone()),
            Unit::Squadron(sq) => {
                internal::Unit::Squadron(squadroninstancesroot[*sq as usize].clone())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Objective {
    ReachNode {
        scalars: internal::ObjectiveScalars,
        node: usize,
    },
    ShipDeath {
        scalars: internal::ObjectiveScalars,
        ship: u64,
    },
    ShipSafe {
        scalars: internal::ObjectiveScalars,
        ship: u64,
        nturns: u64,
    },
    SquadronDeath {
        scalars: internal::ObjectiveScalars,
        squadron: u64,
    },
    SquadronSafe {
        scalars: internal::ObjectiveScalars,
        squadron: u64,
        nturns: u64,
        strengthfraction: f32,
    },
    NodeCapture {
        scalars: internal::ObjectiveScalars,
        node: usize,
    },
    NodeSafe {
        scalars: internal::ObjectiveScalars,
        node: usize,
        nturns: u64,
    },
    SystemCapture {
        scalars: internal::ObjectiveScalars,
        system: usize,
    },
    SystemSafe {
        scalars: internal::ObjectiveScalars,
        system: usize,
        nturns: u64,
        nodesfraction: f32,
    },
}

impl Objective {
    pub fn desiccate(self_entity: &internal::Objective) -> Objective {
        match self_entity {
            internal::Objective::ReachNode { scalars, node } => Objective::ReachNode {
                scalars: *scalars,
                node: node.id,
            },
            internal::Objective::ShipDeath { scalars, ship } => Objective::ShipDeath {
                scalars: *scalars,
                ship: ship.id,
            },
            internal::Objective::ShipSafe {
                scalars,
                ship,
                nturns,
            } => Objective::ShipSafe {
                scalars: *scalars,
                ship: ship.id,
                nturns: *nturns,
            },
            internal::Objective::SquadronDeath { scalars, squadron } => Objective::SquadronDeath {
                scalars: *scalars,
                squadron: squadron.id,
            },
            internal::Objective::SquadronSafe {
                scalars,
                squadron,
                nturns,
                strengthfraction,
            } => Objective::SquadronSafe {
                scalars: *scalars,
                squadron: squadron.id,
                nturns: *nturns,
                strengthfraction: strengthfraction.clone(),
            },
            internal::Objective::NodeCapture { scalars, node } => Objective::NodeCapture {
                scalars: *scalars,
                node: node.id,
            },
            internal::Objective::NodeSafe {
                scalars,
                node,
                nturns,
            } => Objective::NodeSafe {
                scalars: *scalars,
                node: node.id,
                nturns: *nturns,
            },
            internal::Objective::SystemCapture { scalars, system } => Objective::SystemCapture {
                scalars: *scalars,
                system: system.id,
            },
            internal::Objective::SystemSafe {
                scalars,
                system,
                nturns,
                nodesfraction,
            } => Objective::SystemSafe {
                scalars: *scalars,
                system: system.id,
                nturns: *nturns,
                nodesfraction: nodesfraction.clone(),
            },
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        systemsroot: &Vec<Arc<internal::System>>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> internal::Objective {
        match self {
            Objective::ReachNode { scalars, node } => internal::Objective::ReachNode {
                scalars: *scalars,
                node: nodesroot[*node].clone(),
            },
            Objective::ShipDeath { scalars, ship } => internal::Objective::ShipDeath {
                scalars: *scalars,
                ship: shipinstancesroot[*ship as usize].clone(),
            },
            Objective::ShipSafe {
                scalars,
                ship,
                nturns,
            } => internal::Objective::ShipSafe {
                scalars: *scalars,
                ship: shipinstancesroot[*ship as usize].clone(),
                nturns: *nturns,
            },
            Objective::SquadronDeath { scalars, squadron } => internal::Objective::SquadronDeath {
                scalars: *scalars,
                squadron: squadroninstancesroot[*squadron as usize].clone(),
            },
            Objective::SquadronSafe {
                scalars,
                squadron,
                nturns,
                strengthfraction,
            } => internal::Objective::SquadronSafe {
                scalars: *scalars,
                squadron: squadroninstancesroot[*squadron as usize].clone(),
                nturns: *nturns,
                strengthfraction: strengthfraction.clone(),
            },
            Objective::NodeCapture { scalars, node } => internal::Objective::NodeCapture {
                scalars: *scalars,
                node: nodesroot[*node].clone(),
            },
            Objective::NodeSafe {
                scalars,
                node,
                nturns,
            } => internal::Objective::NodeSafe {
                scalars: *scalars,
                node: nodesroot[*node].clone(),
                nturns: *nturns,
            },
            Objective::SystemCapture { scalars, system } => internal::Objective::SystemCapture {
                scalars: *scalars,
                system: systemsroot[*system].clone(),
            },
            Objective::SystemSafe {
                scalars,
                system,
                nturns,
                nodesfraction,
            } => internal::Objective::SystemSafe {
                scalars: *scalars,
                system: systemsroot[*system].clone(),
                nturns: *nturns,
                nodesfraction: nodesfraction.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub visiblename: String,
    pub objectives: Vec<Objective>,
}

impl Operation {
    pub fn desiccate(self_entity: &internal::Operation) -> Operation {
        Operation {
            visiblename: self_entity.visiblename.clone(),
            objectives: self_entity
                .objectives
                .iter()
                .map(|x| Objective::desiccate(x))
                .collect(),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        systemsroot: &Vec<Arc<internal::System>>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> internal::Operation {
        internal::Operation {
            visiblename: self.visiblename.clone(),
            objectives: self
                .objectives
                .iter()
                .map(|x| {
                    x.rehydrate(
                        &nodesroot,
                        &systemsroot,
                        &shipinstancesroot,
                        &squadroninstancesroot,
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct FactionForces {
    pub local_forces: Vec<Unit>,
    pub reinforcements: Vec<(u64, Vec<Unit>)>,
}

impl FactionForces {
    pub fn desiccate(self_entity: &internal::FactionForces) -> FactionForces {
        FactionForces {
            local_forces: self_entity
                .local_forces
                .iter()
                .map(|x| Unit::desiccate(x))
                .collect(),
            reinforcements: self_entity
                .reinforcements
                .iter()
                .map(|(distance, units)| {
                    (
                        *distance,
                        units.iter().map(|x| Unit::desiccate(x)).collect(),
                    )
                })
                .collect(),
        }
    }
    pub fn rehydrate(
        &self,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
    ) -> internal::FactionForces {
        internal::FactionForces {
            local_forces: self
                .local_forces
                .iter()
                .map(|x| x.rehydrate(&shipinstancesroot, &squadroninstancesroot))
                .collect(),
            reinforcements: self
                .reinforcements
                .iter()
                .map(|(distance, units)| {
                    (
                        *distance,
                        units
                            .iter()
                            .map(|x| x.rehydrate(&shipinstancesroot, &squadroninstancesroot))
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnitStatus {
    pub location: Option<UnitLocation>,
    pub damage: u64,
    pub engine_damage: Vec<u64>,
}

impl UnitStatus {
    pub fn desiccate(self_entity: &internal::UnitStatus) -> UnitStatus {
        UnitStatus {
            location: self_entity
                .location
                .clone()
                .map(|x| UnitLocation::desiccate(&x)),
            damage: self_entity.damage,
            engine_damage: self_entity.engine_damage.clone(),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
        hangarslist: &Vec<Arc<internal::HangarInstance>>,
    ) -> internal::UnitStatus {
        internal::UnitStatus {
            location: self
                .location
                .clone()
                .map(|x| x.rehydrate(&nodesroot, &squadroninstancesroot, &hangarslist)),
            damage: self.damage,
            engine_damage: self.engine_damage.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Engagement {
    pub visiblename: String,
    pub turn: u64,
    pub coalitions: HashMap<u64, HashMap<usize, FactionForces>>,
    pub aggressor: Option<usize>,
    pub objectives: HashMap<usize, Vec<Objective>>,
    pub location: usize,
    pub duration: u64,
    pub victors: (usize, u64),
    pub unit_status: HashMap<u64, HashMap<usize, HashMap<Unit, UnitStatus>>>,
}

impl Engagement {
    pub fn desiccate(self_entity: &internal::Engagement) -> Engagement {
        Engagement {
            visiblename: self_entity.visiblename.clone(),
            turn: self_entity.turn,
            coalitions: self_entity
                .coalitions
                .iter()
                .map(|(index, faction_map)| {
                    (
                        *index,
                        faction_map
                            .iter()
                            .map(|(faction, forces)| (faction.id, FactionForces::desiccate(forces)))
                            .collect(),
                    )
                })
                .collect(),
            aggressor: self_entity.aggressor.clone().map(|x| x.id),
            objectives: self_entity
                .objectives
                .iter()
                .map(|(faction, objs)| {
                    (
                        faction.id,
                        objs.iter().map(|x| Objective::desiccate(x)).collect(),
                    )
                })
                .collect(),
            location: self_entity.location.id,
            duration: self_entity.duration,
            victors: (self_entity.victors.0.id, self_entity.victors.1),
            unit_status: self_entity
                .unit_status
                .iter()
                .map(|(index, faction_map)| {
                    (
                        *index,
                        faction_map
                            .iter()
                            .map(|(faction, unit_map)| {
                                (
                                    faction.id,
                                    unit_map
                                        .iter()
                                        .map(|(u, us)| {
                                            (Unit::desiccate(u), UnitStatus::desiccate(us))
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<internal::Node>>,
        systemsroot: &Vec<Arc<internal::System>>,
        factionsroot: &Vec<Arc<internal::Faction>>,
        shipinstancesroot: &Vec<Arc<internal::ShipInstance>>,
        squadroninstancesroot: &Vec<Arc<internal::SquadronInstance>>,
        hangarslist: &Vec<Arc<internal::HangarInstance>>,
    ) -> internal::Engagement {
        internal::Engagement {
            visiblename: self.visiblename.clone(),
            turn: self.turn,
            coalitions: self
                .coalitions
                .iter()
                .map(|(index, faction_map)| {
                    (
                        *index,
                        faction_map
                            .iter()
                            .map(|(faction, forces)| {
                                (
                                    factionsroot[*faction].clone(),
                                    forces.rehydrate(&shipinstancesroot, &squadroninstancesroot),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            aggressor: self.aggressor.clone().map(|x| factionsroot[x].clone()),
            objectives: self
                .objectives
                .iter()
                .map(|(faction, objs)| {
                    (
                        factionsroot[*faction].clone(),
                        objs.iter()
                            .map(|x| {
                                x.rehydrate(
                                    &nodesroot,
                                    &systemsroot,
                                    &shipinstancesroot,
                                    &squadroninstancesroot,
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            location: nodesroot[self.location].clone(),
            duration: self.duration,
            victors: (factionsroot[self.victors.0].clone(), self.victors.1),
            unit_status: self
                .unit_status
                .iter()
                .map(|(index, faction_map)| {
                    (
                        *index,
                        faction_map
                            .iter()
                            .map(|(faction, unit_map)| {
                                (
                                    factionsroot[*faction].clone(),
                                    unit_map
                                        .iter()
                                        .map(|(u, us)| {
                                            (
                                                u.rehydrate(
                                                    &shipinstancesroot,
                                                    &squadroninstancesroot,
                                                ),
                                                us.rehydrate(
                                                    &nodesroot,
                                                    &squadroninstancesroot,
                                                    &hangarslist,
                                                ),
                                            )
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalSalience {
    pub factionsalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub resourcesalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub unitclasssalience: Vec<Vec<Vec<[f32; 2]>>>,
}

impl GlobalSalience {
    fn desiccate(self_entity: &internal::GlobalSalience) -> GlobalSalience {
        GlobalSalience {
            factionsalience: self_entity.factionsalience.read().unwrap().clone(),
            resourcesalience: self_entity.resourcesalience.read().unwrap().clone(),
            unitclasssalience: self_entity.unitclasssalience.read().unwrap().clone(),
        }
    }
    fn rehydrate(&self) -> internal::GlobalSalience {
        internal::GlobalSalience {
            factionsalience: RwLock::new(self.factionsalience.clone()),
            resourcesalience: RwLock::new(self.resourcesalience.clone()),
            unitclasssalience: RwLock::new(self.unitclasssalience.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Root {
    pub config: internal::Config,
    pub nodeflavors: Vec<internal::NodeFlavor>,
    pub nodes: Vec<Node>,
    pub systems: Vec<System>,
    pub edgeflavors: Vec<internal::EdgeFlavor>,
    pub edges: HashMap<(usize, usize), usize>,
    pub neighbors: HashMap<usize, Vec<usize>>,
    pub factions: Vec<internal::Faction>,
    pub wars: HashSet<(usize, usize)>,
    pub resources: Vec<internal::Resource>,
    pub hangarclasses: Vec<internal::HangarClass>,
    pub hangarinstancecounter: u64,
    pub engineclasses: Vec<EngineClass>,
    pub repairerclasses: Vec<RepairerClass>,
    pub factoryclasses: Vec<FactoryClass>,
    pub shipyardclasses: Vec<ShipyardClass>,
    pub shipais: Vec<ShipAI>,
    pub shipflavors: Vec<internal::ShipFlavor>,
    pub squadronflavors: Vec<internal::SquadronFlavor>,
    pub shipclasses: Vec<ShipClass>,
    pub squadronclasses: Vec<SquadronClass>,
    pub shipinstances: Vec<ShipInstance>,
    pub squadroninstances: Vec<SquadronInstance>,
    pub unitcounter: u64,
    pub engagements: Vec<Engagement>,
    pub globalsalience: GlobalSalience,
    pub turn: u64,
}

impl Root {
    pub fn desiccate(self_entity: &internal::Root) -> Root {
        Root {
            config: self_entity.config.clone(),
            nodeflavors: self_entity
                .nodeflavors
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            nodes: self_entity
                .nodes
                .iter()
                .map(|x| Node::desiccate(x))
                .collect(),
            systems: self_entity
                .systems
                .iter()
                .map(|x| System::desiccate(x))
                .collect(),
            edgeflavors: self_entity
                .edgeflavors
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            edges: self_entity
                .edges
                .iter()
                .map(|((n1, n2), flavor)| ((n1.id, n2.id), flavor.id))
                .collect(),
            neighbors: self_entity
                .neighbors
                .iter()
                .map(|(node, nodes)| (node.id, nodes.iter().map(|rhs| rhs.id).collect()))
                .collect(),
            factions: self_entity
                .factions
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            wars: self_entity
                .wars
                .iter()
                .map(|(f1, f2)| (f1.id, f2.id))
                .collect(),
            resources: self_entity
                .resources
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            hangarclasses: self_entity
                .hangarclasses
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            hangarinstancecounter: self_entity
                .hangarinstancecounter
                .load(atomic::Ordering::Relaxed),
            engineclasses: self_entity
                .engineclasses
                .iter()
                .map(|x| EngineClass::desiccate(x))
                .collect(),
            repairerclasses: self_entity
                .repairerclasses
                .iter()
                .map(|x| RepairerClass::desiccate(x))
                .collect(),
            factoryclasses: self_entity
                .factoryclasses
                .iter()
                .map(|x| FactoryClass::desiccate(x))
                .collect(),
            shipyardclasses: self_entity
                .shipyardclasses
                .iter()
                .map(|x| ShipyardClass::desiccate(x))
                .collect(),
            shipais: self_entity
                .shipais
                .iter()
                .map(|x| ShipAI::desiccate(x))
                .collect(),
            shipflavors: self_entity
                .shipflavors
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            squadronflavors: self_entity
                .squadronflavors
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
                .collect(),
            shipclasses: self_entity
                .shipclasses
                .iter()
                .map(|x| ShipClass::desiccate(x))
                .collect(),
            squadronclasses: self_entity
                .squadronclasses
                .iter()
                .map(|x| SquadronClass::desiccate(x))
                .collect(),
            shipinstances: self_entity
                .shipinstances
                .read()
                .unwrap()
                .iter()
                .map(|x| ShipInstance::desiccate(x))
                .collect(),
            squadroninstances: self_entity
                .squadroninstances
                .read()
                .unwrap()
                .iter()
                .map(|x| SquadronInstance::desiccate(x))
                .collect(),
            unitcounter: self_entity.unitcounter.load(atomic::Ordering::Relaxed),
            engagements: self_entity
                .engagements
                .read()
                .unwrap()
                .iter()
                .map(|x| Engagement::desiccate(x))
                .collect(),
            globalsalience: GlobalSalience::desiccate(&self_entity.globalsalience),
            turn: self_entity.turn.load(atomic::Ordering::Relaxed),
        }
    }
    pub fn rehydrate(mut self) -> internal::Root {
        let config = self.config.clone();
        let nodeflavors = self.nodeflavors.drain(0..).map(|x| Arc::new(x)).collect();
        let edgeflavors = self.edgeflavors.drain(0..).map(|x| Arc::new(x)).collect();
        let factions: Vec<_> = self.factions.drain(0..).map(|x| Arc::new(x)).collect();
        let wars = self
            .wars
            .iter()
            .map(|(f1, f2)| (factions[*f1].clone(), factions[*f2].clone()))
            .collect();
        let resources = self.resources.drain(0..).map(|x| Arc::new(x)).collect();
        let hangarclasses = self.hangarclasses.drain(0..).map(|x| Arc::new(x)).collect();
        let hangarinstancecounter = Arc::new(AtomicU64::new(self.hangarinstancecounter));
        let engineclasses = self
            .engineclasses
            .drain(0..)
            .map(|x| Arc::new(x.rehydrate(&nodeflavors, &edgeflavors, &resources)))
            .collect();
        let repairerclasses = self
            .repairerclasses
            .drain(0..)
            .map(|x| Arc::new(x.rehydrate(&resources)))
            .collect();
        let factoryclasses = self
            .factoryclasses
            .drain(0..)
            .map(|x| Arc::new(x.rehydrate(&resources)))
            .collect();
        let shipyardclasses = self
            .shipyardclasses
            .drain(0..)
            .map(|x| Arc::new(x.rehydrate(&resources)))
            .collect();
        let shipais = self
            .shipais
            .drain(0..)
            .map(|x| Arc::new(x.rehydrate(&resources)))
            .collect();
        let shipflavors = self.shipflavors.drain(0..).map(|x| Arc::new(x)).collect();
        let squadronflavors = self
            .squadronflavors
            .drain(0..)
            .map(|x| Arc::new(x))
            .collect();
        let shipclasses = self
            .shipclasses
            .iter()
            .map(|x| {
                Arc::new(x.rehydrate(
                    &factions,
                    &resources,
                    &hangarclasses,
                    &engineclasses,
                    &repairerclasses,
                    &factoryclasses,
                    &shipyardclasses,
                    &shipais,
                    &shipflavors,
                ))
            })
            .collect();
        let squadronclasses = self
            .squadronclasses
            .iter()
            .map(|x| Arc::new(x.rehydrate(&factions, &squadronflavors)))
            .collect();
        let nodes = self
            .nodes
            .iter()
            .map(|x| {
                Arc::new(x.rehydrate(
                    &nodeflavors,
                    &factions,
                    &resources,
                    &factoryclasses,
                    &shipyardclasses,
                    &shipclasses,
                ))
            })
            .collect();
        let systems = self
            .systems
            .iter()
            .map(|x| Arc::new(x.rehydrate(&nodes)))
            .collect();
        let edges = self
            .edges
            .drain()
            .map(|((n1, n2), flavor)| {
                (
                    (nodes[n1].clone(), nodes[n2].clone()),
                    edgeflavors[flavor].clone(),
                )
            })
            .collect();
        let neighbors = self
            .neighbors
            .drain()
            .map(|(node, neighs)| {
                (
                    nodes[node].clone(),
                    neighs.iter().map(|rhs| nodes[*rhs].clone()).collect(),
                )
            })
            .collect();
        let shipinstances: RwLock<Vec<Arc<internal::ShipInstance>>> = RwLock::new(
            self.shipinstances
                .iter()
                .map(|x| {
                    Arc::new(x.rehydrate(
                        &nodeflavors,
                        &nodes,
                        &edgeflavors,
                        &factions,
                        &resources,
                        &engineclasses,
                        &repairerclasses,
                        &factoryclasses,
                        &shipyardclasses,
                        &shipais,
                        &shipclasses,
                    ))
                })
                .collect(),
        );
        let squadroninstances = RwLock::new(
            self.squadroninstances
                .drain(0..)
                .map(|x| Arc::new(x.rehydrate(&nodes, &factions, &squadronclasses)))
                .collect(),
        );
        //here we go through and add the units to the nodes, now that we have the data for unit instances
        nodes.iter().for_each(|node| {
            Node::add_units(
                node,
                &self.nodes,
                &shipinstances.read().unwrap(),
                &squadroninstances.read().unwrap(),
            )
        });
        let hangarslist = shipinstances
            .read()
            .unwrap()
            .iter()
            .map(|shipinstance| {
                ShipInstance::add_hangars_and_objectives(
                    shipinstance,
                    &self.shipinstances,
                    &nodes,
                    &systems,
                    &hangarclasses,
                    &shipinstances.read().unwrap(),
                    &squadroninstances.read().unwrap(),
                )
            })
            .flatten()
            .collect();
        shipinstances.read().unwrap().iter().for_each(|ship| {
            ShipInstance::set_location(
                ship,
                &self.shipinstances,
                &nodes,
                &squadroninstances.read().unwrap(),
                &hangarslist,
            )
        });
        squadroninstances
            .read()
            .unwrap()
            .iter()
            .for_each(|squadron| {
                SquadronInstance::add_daughters_and_objectives_set_location(
                    squadron,
                    &self.squadroninstances,
                    &nodes,
                    &systems,
                    &shipinstances.read().unwrap(),
                    &squadroninstances.read().unwrap(),
                    &hangarslist,
                )
            });
        let unitcounter = Arc::new(AtomicU64::new(self.unitcounter));
        let engagements = RwLock::new(
            self.engagements
                .drain(0..)
                .map(|x| {
                    Arc::new(x.rehydrate(
                        &nodes,
                        &systems,
                        &factions,
                        &shipinstances.read().unwrap(),
                        &squadroninstances.read().unwrap(),
                        &hangarslist,
                    ))
                })
                .collect(),
        );
        let globalsalience = self.globalsalience.rehydrate();
        let turn = Arc::new(AtomicU64::new(self.turn));

        internal::Root {
            config,
            nodeflavors,
            nodes,
            systems,
            edgeflavors,
            edges,
            neighbors,
            factions,
            wars,
            resources,
            hangarclasses,
            hangarinstancecounter,
            engineclasses,
            repairerclasses,
            factoryclasses,
            shipyardclasses,
            shipais,
            shipflavors,
            squadronflavors,
            shipclasses,
            squadronclasses,
            shipinstances,
            squadroninstances,
            unitcounter,
            engagements,
            globalsalience,
            turn,
        }
    }
}
