//this is the section of the program that converts data to or from a format suitable for transmission to or reciept from other programs
use crate::internal;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64};
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct System {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub nodes: Vec<usize>,
}

impl System {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluripotentStockpile {
    pub visibility: bool,
    pub contents: HashMap<usize, u64>,
    pub allowed: Option<Vec<usize>>,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl PluripotentStockpile {
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

#[derive(Debug, Serialize, Deserialize)]
pub struct SharedStockpile {
    pub resourcetype: usize,
    pub contents: u64,
    pub rate: u64,
    pub capacity: u64,
}

impl SharedStockpile {
    fn rehydrate(&self, resourcesroot: &Vec<Arc<internal::Resource>>) -> internal::SharedStockpile {
        internal::SharedStockpile {
            resourcetype: resourcesroot[self.resourcetype].clone(),
            contents: Arc::new(AtomicU64::new(self.contents)),
            rate: self.rate,
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HangarInstanceMut {
    pub visibility: bool,
    pub contents: Vec<Unit>,
}

impl HangarInstanceMut {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HangarInstance {
    pub id: u64,
    pub class: usize,
    pub mother: u64,
    pub mutables: HangarInstanceMut,
}

impl HangarInstance {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct ShipInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: usize, //which class of ship this is
    pub mutables: ShipInstanceMut,
}

impl ShipInstance {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquadronInstanceMut {
    pub visibility: bool,
    pub location: UnitLocation,
    pub daughters: Vec<Unit>,
    pub allegiance: usize,
    pub objectives: Vec<Objective>,
    pub ghost: bool,
}

impl SquadronInstanceMut {
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

#[derive(Debug, Serialize, Deserialize)]
pub struct SquadronInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: usize,
    pub idealstrength: u64,
    pub mutables: SquadronInstanceMut,
}

impl SquadronInstance {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Operation {
    pub visiblename: String,
    pub objectives: Vec<Objective>,
}

impl Operation {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionForces {
    pub local_forces: Vec<Unit>,
    pub reinforcements: Vec<(u64, Vec<Unit>)>,
}

impl FactionForces {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitStatus {
    pub location: Option<UnitLocation>,
    pub damage: u64,
    pub engine_damage: Vec<u64>,
}

impl UnitStatus {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalSalience {
    pub factionsalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub resourcesalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub unitclasssalience: Vec<Vec<Vec<[f32; 2]>>>,
}

impl GlobalSalience {
    fn rehydrate(&self) -> internal::GlobalSalience {
        internal::GlobalSalience {
            factionsalience: RwLock::new(self.factionsalience.clone()),
            resourcesalience: RwLock::new(self.resourcesalience.clone()),
            unitclasssalience: RwLock::new(self.unitclasssalience.clone()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
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
