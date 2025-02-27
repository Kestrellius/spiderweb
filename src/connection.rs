//this is the section of the program that converts data to or from a format suitable for transmission to or reciept from other programs
use crate::internal::export;
use serde::{Deserialize, Serialize};
use serde_json_any_key::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Debug, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct UnitContainer {
    contents: Vec<Unit>,
}

impl UnitContainer {
    fn desiccate(self_entity: &export::UnitContainer) -> UnitContainer {
        UnitContainer {
            contents: self_entity
                .contents
                .iter()
                .map(|daughter| Unit::desiccate(daughter))
                .collect(),
        }
    }
    fn rehydrate(
        &self,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> export::UnitContainer {
        export::UnitContainer {
            contents: self
                .contents
                .iter()
                .map(|x| x.rehydrate(shipsroot, squadronsroot))
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct NodeMut {
    pub visibility: bool,
    pub flavor: usize, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub factory_list: Vec<Factory>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyardlist: Vec<Shipyard>,
    pub allegiance: usize, //faction that currently holds the node
    pub efficiency: f32, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub balance_stockpiles: bool,
    pub balance_hangars: bool,
    pub check_for_battles: bool,
    pub stockpiles_balanced: bool,
    pub hangars_balanced: bool,
}

impl NodeMut {
    fn desiccate(self_entity: &export::NodeMut) -> NodeMut {
        NodeMut {
            visibility: self_entity.visibility,
            flavor: self_entity.flavor.id,
            factory_list: self_entity
                .factories
                .iter()
                .map(|x| Factory::desiccate(x))
                .collect(),
            shipyardlist: self_entity
                .shipyards
                .iter()
                .map(|x| Shipyard::desiccate(x))
                .collect(),
            allegiance: self_entity.allegiance.id,
            efficiency: self_entity.efficiency,
            balance_stockpiles: self_entity.transact_resources,
            balance_hangars: self_entity.transact_units,
            check_for_battles: self_entity.check_for_battles,
            stockpiles_balanced: self_entity.resources_transacted,
            hangars_balanced: self_entity.units_transacted,
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
        factoryclassesroot: &Vec<Arc<export::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<export::ShipyardClass>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
    ) -> export::NodeMut {
        export::NodeMut {
            visibility: self.visibility,
            flavor: nodeflavorsroot[self.flavor].clone(),
            factories: self
                .factory_list
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &factoryclassesroot))
                .collect(),
            shipyards: self
                .shipyardlist
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &shipyardclassesroot, &shipclassesroot))
                .collect(),
            allegiance: factionsroot[self.allegiance].clone(),
            efficiency: self.efficiency,
            transact_resources: self.balance_stockpiles,
            transact_units: self.balance_hangars,
            check_for_battles: self.check_for_battles,
            resources_transacted: self.stockpiles_balanced,
            units_transacted: self.hangars_balanced,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub visible_name: String, //location name as shown to player
    pub position: [i64; 3], //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    pub description: String,
    pub environment: String, //name of the FRED environment to use for missions set in this node
    pub bitmap: Option<(String, f32)>, //name of the bitmap used to depict this node in other nodes' FRED environments, and a size factor
    pub mutables: NodeMut,
    pub unit_container: UnitContainer,
}

impl Node {
    fn desiccate(self_entity: &export::Node) -> Node {
        Node {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            position: self_entity.position,
            description: self_entity.description.clone(),
            environment: self_entity.environment.clone(),
            bitmap: self_entity.bitmap.clone(),
            mutables: NodeMut::desiccate(&self_entity.mutables.read().unwrap()),
            unit_container: UnitContainer::desiccate(&self_entity.unit_container.read().unwrap()),
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
        factoryclassesroot: &Vec<Arc<export::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<export::ShipyardClass>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
    ) -> export::Node {
        export::Node {
            id: self.id,
            visible_name: self.visible_name.clone(),
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
            unit_container: RwLock::new(export::UnitContainer::new()),
        }
    }
    fn add_units(
        node: &Arc<export::Node>,
        connectionnodes: &Vec<Node>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) {
        node.unit_container.write().unwrap().contents = connectionnodes[node.id]
            .unit_container
            .contents
            .iter()
            .map(|x| x.rehydrate(&shipsroot, &squadronsroot))
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
pub struct Cluster {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub nodes: Vec<usize>,
}

impl Cluster {
    fn desiccate(self_entity: &export::Cluster) -> Cluster {
        Cluster {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility.clone(),
            nodes: self_entity.nodes.iter().map(|x| x.id).collect(),
        }
    }
    fn rehydrate(&self, nodesroot: &Vec<Arc<export::Node>>) -> export::Cluster {
        export::Cluster {
            id: self.id,
            visible_name: self.visible_name.clone(),
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
    pub resource_type: usize,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl UnipotentStockpile {
    fn desiccate(self_entity: &export::UnipotentStockpile) -> UnipotentStockpile {
        UnipotentStockpile {
            visibility: self_entity.visibility,
            resource_type: self_entity.resource_type.id,
            contents: self_entity.contents,
            rate: self_entity.rate,
            target: self_entity.target,
            capacity: self_entity.capacity,
            propagates: self_entity.propagates,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<export::Resource>>) -> export::UnipotentStockpile {
        export::UnipotentStockpile {
            visibility: self.visibility,
            resource_type: resourcesroot[self.resource_type].clone(),
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
    fn desiccate(self_entity: &export::PluripotentStockpile) -> PluripotentStockpile {
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
        resourcesroot: &Vec<Arc<export::Resource>>,
    ) -> export::PluripotentStockpile {
        export::PluripotentStockpile {
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
    pub resource_type: usize,
    pub contents: u64,
    pub rate: u64,
    pub capacity: u64,
}

impl SharedStockpile {
    fn desiccate(self_entity: &export::SharedStockpile) -> SharedStockpile {
        SharedStockpile {
            resource_type: self_entity.resource_type.id,
            contents: self_entity.contents.load(atomic::Ordering::Relaxed),
            rate: self_entity.rate,
            capacity: self_entity.capacity,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<export::Resource>>) -> export::SharedStockpile {
        export::SharedStockpile {
            resource_type: resourcesroot[self.resource_type].clone(),
            contents: Arc::new(AtomicU64::new(self.contents)),
            rate: self.rate,
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HangarMut {
    pub visibility: bool,
}

impl HangarMut {
    fn desiccate(self_entity: &export::HangarMut) -> HangarMut {
        HangarMut {
            visibility: self_entity.visibility,
        }
    }
    fn rehydrate(&self) -> export::HangarMut {
        export::HangarMut {
            visibility: self.visibility,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Hangar {
    pub id: u64,
    pub class: usize,
    pub mother: u64,
    pub mutables: HangarMut,
    pub unit_container: UnitContainer,
}

impl Hangar {
    pub fn desiccate(self_entity: &export::Hangar) -> Hangar {
        Hangar {
            id: self_entity.id,
            class: self_entity.class.id,
            mother: self_entity.mother.id,
            mutables: HangarMut::desiccate(&self_entity.mutables.read().unwrap()),
            unit_container: UnitContainer::desiccate(&self_entity.unit_container.read().unwrap()),
        }
    }
    pub fn rehydrate(
        &self,
        hangarclassesroot: &Vec<Arc<export::HangarClass>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> export::Hangar {
        export::Hangar {
            id: self.id,
            class: hangarclassesroot[self.class].clone(),
            mother: shipsroot
                .iter()
                .find(|ship| ship.id == self.mother)
                .unwrap()
                .clone(),
            mutables: RwLock::new(self.mutables.rehydrate()),
            unit_container: RwLock::new(self.unit_container.rehydrate(&shipsroot, &squadronsroot)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub base_health: Option<u64>,
    pub toughness_scalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<usize>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<usize>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl EngineClass {
    pub fn desiccate(self_entity: &export::EngineClass) -> EngineClass {
        EngineClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            base_health: self_entity.base_health,
            toughness_scalar: self_entity.toughness_scalar.clone(),
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
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        edgeflavorsroot: &Vec<Arc<export::EdgeFlavor>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
    ) -> export::EngineClass {
        export::EngineClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
            base_health: self.base_health,
            toughness_scalar: self.toughness_scalar.clone(),
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
pub struct Engine {
    pub engineclass: usize,
    pub visibility: bool,
    pub health: Option<u64>,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<usize>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<usize>, //the engine won't allow a ship to traverse edges of these flavors
    pub last_move_turn: u64,
}

impl Engine {
    fn desiccate(self_entity: &export::Engine) -> Engine {
        Engine {
            engineclass: self_entity.class.id,
            visibility: self_entity.visibility,
            health: self_entity.health,
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
            last_move_turn: self_entity.last_move_turn,
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        edgeflavorsroot: &Vec<Arc<export::EdgeFlavor>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
        engineclassesroot: &Vec<Arc<export::EngineClass>>,
    ) -> export::Engine {
        export::Engine {
            class: engineclassesroot[self.engineclass].clone(),
            visibility: self.visibility,
            health: self.health,
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
            last_move_turn: self.last_move_turn,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairerClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
    pub strategic_weapon_repair_points: i64,
    pub strategic_weapon_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerClass {
    fn desiccate(self_entity: &export::RepairerClass) -> RepairerClass {
        RepairerClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            repair_points: self_entity.repair_points,
            repair_factor: self_entity.repair_factor,
            engine_repair_points: self_entity.engine_repair_points,
            engine_repair_factor: self_entity.engine_repair_factor,
            strategic_weapon_repair_points: self_entity.subsystem_repair_points,
            strategic_weapon_repair_factor: self_entity.subsystem_repair_factor,
            per_engagement: self_entity.per_engagement,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<export::Resource>>) -> export::RepairerClass {
        export::RepairerClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
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
            subsystem_repair_points: self.strategic_weapon_repair_points,
            subsystem_repair_factor: self.strategic_weapon_repair_factor,
            per_engagement: self.per_engagement,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Repairer {
    pub repairerclass: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
}

impl Repairer {
    fn desiccate(self_entity: &export::Repairer) -> Repairer {
        Repairer {
            repairerclass: self_entity.class.id,
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<export::Resource>>,
        repairerclassesroot: &Vec<Arc<export::RepairerClass>>,
    ) -> export::Repairer {
        export::Repairer {
            class: repairerclassesroot[self.repairerclass].clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategicWeaponClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<usize>, //the weapon won't fire into nodes of these flavors
    pub forbidden_edgeflavors: Vec<usize>, //the weapon won't fire across edges of these flavors
    pub damage: ((i64, i64), (f32, f32)),  //lower and upper bounds for damage done by a single shot
    pub engine_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to engine done by a single shot
    pub strategic_weapon_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to strategic weapon done by a single shot
    pub accuracy: f32, //divided by target's strategicweaponevasionscalar to get hit probability as a fraction of 1.0
    pub range: u64,    //how many edges away the weapon can reach
    pub shots: (u64, u64), //lower and upper bounds for maximum number of shots the weapon fires each time it's activated
    pub targets_enemies: bool,
    pub targets_allies: bool,
    pub targets_neutrals: bool,
    pub target_relations_lower_bound: Option<f32>,
    pub target_relations_upper_bound: Option<f32>,
    pub target_priorities_class: HashMap<export::ShipClassID, f32>, //how strongly weapon will prioritize ships of each class; classes absent from list will default to 1.0
    pub target_priorities_flavor: HashMap<usize, f32>, //how strongly weapon will prioritize ships of each flavor; flavors absent from list will default to 1.0
}

impl StrategicWeaponClass {
    fn desiccate(self_entity: &export::StrategicWeaponClass) -> StrategicWeaponClass {
        StrategicWeaponClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
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
            damage: self_entity.damage,
            engine_damage: self_entity.engine_damage,
            strategic_weapon_damage: self_entity.strategic_weapon_damage,
            accuracy: self_entity.accuracy,
            range: self_entity.range,
            shots: self_entity.shots,
            targets_enemies: self_entity.targets_enemies,
            targets_allies: self_entity.targets_allies,
            targets_neutrals: self_entity.targets_neutrals,
            target_relations_lower_bound: self_entity.target_relations_lower_bound,
            target_relations_upper_bound: self_entity.target_relations_upper_bound,
            target_priorities_class: self_entity.target_priorities_class.clone(),
            target_priorities_flavor: self_entity
                .target_priorities_flavor
                .iter()
                .map(|(flavor, val)| (flavor.id, *val))
                .collect(),
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<export::Resource>>,
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        edgeflavorsroot: &Vec<Arc<export::EdgeFlavor>>,
        shipflavorsroot: &Vec<Arc<export::ShipFlavor>>,
    ) -> export::StrategicWeaponClass {
        export::StrategicWeaponClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
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
            damage: self.damage,
            engine_damage: self.engine_damage,
            strategic_weapon_damage: self.strategic_weapon_damage,
            accuracy: self.accuracy,
            range: self.range,
            shots: self.shots,
            targets_enemies: self.targets_enemies,
            targets_allies: self.targets_allies,
            targets_neutrals: self.targets_neutrals,
            target_relations_lower_bound: self.target_relations_lower_bound,
            target_relations_upper_bound: self.target_relations_upper_bound,
            target_priorities_class: self.target_priorities_class.clone(),
            target_priorities_flavor: self
                .target_priorities_flavor
                .iter()
                .map(|(flavor, val)| (shipflavorsroot[*flavor].clone(), *val))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategicWeapon {
    pub class: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
}

impl StrategicWeapon {
    fn desiccate(self_entity: &export::StrategicWeapon) -> StrategicWeapon {
        StrategicWeapon {
            class: self_entity.class.id,
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<export::Resource>>,
        strategicweaponclassesroot: &Vec<Arc<export::StrategicWeaponClass>>,
    ) -> export::StrategicWeapon {
        export::StrategicWeapon {
            class: strategicweaponclassesroot[self.class].clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(resourcesroot))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoryClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    fn desiccate(self_entity: &export::FactoryClass) -> FactoryClass {
        FactoryClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
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
    fn rehydrate(&self, resourcesroot: &Vec<Arc<export::Resource>>) -> export::FactoryClass {
        export::FactoryClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
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
pub struct Factory {
    //this is an actual factory, derived from a factory class
    pub factoryclass: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl Factory {
    fn desiccate(self_entity: &export::Factory) -> Factory {
        Factory {
            factoryclass: self_entity.class.id,
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
        resourcesroot: &Vec<Arc<export::Resource>>,
        factoryclassesroot: &Vec<Arc<export::FactoryClass>>,
    ) -> export::Factory {
        export::Factory {
            class: factoryclassesroot[self.factoryclass].clone(),
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
    pub visible_name: Option<String>,
    pub description: Option<String>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<export::ShipClassID, u64>,
    pub constructrate: u64,
    pub efficiency: f32,
}

impl ShipyardClass {
    fn desiccate(self_entity: &export::ShipyardClass) -> ShipyardClass {
        ShipyardClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            visibility: self_entity.visibility,
            inputs: self_entity
                .inputs
                .iter()
                .map(|x| UnipotentStockpile::desiccate(x))
                .collect(),
            outputs: self_entity.outputs.clone(),
            constructrate: self_entity.construct_rate,
            efficiency: self_entity.efficiency,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<export::Resource>>) -> export::ShipyardClass {
        export::ShipyardClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            visibility: self.visibility,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            outputs: self.outputs.clone(),
            construct_rate: self.constructrate,
            efficiency: self.efficiency,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shipyard {
    pub class: usize,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<usize, u64>,
    pub constructpoints: u64,
    pub efficiency: f32,
}

impl Shipyard {
    fn desiccate(self_entity: &export::Shipyard) -> Shipyard {
        Shipyard {
            class: self_entity.class.id,
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
            constructpoints: self_entity.construct_points,
            efficiency: self_entity.efficiency,
        }
    }
    fn rehydrate(
        &self,
        resourcesroot: &Vec<Arc<export::Resource>>,
        shipyardclassesroot: &Vec<Arc<export::ShipyardClass>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
    ) -> export::Shipyard {
        export::Shipyard {
            class: shipyardclassesroot[self.class].clone(),
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
            construct_points: self.constructpoints,
            efficiency: self.efficiency,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Subsystem {
    pub class: usize,
    pub visibility: bool,
    pub health: Option<u64>,
}

impl Subsystem {
    fn desiccate(self_entity: &export::Subsystem) -> Subsystem {
        Subsystem {
            class: self_entity.class.id,
            visibility: self_entity.visibility,
            health: self_entity.health,
        }
    }
    fn rehydrate(
        &self,
        subsystemclassesroot: &Vec<Arc<export::SubsystemClass>>,
    ) -> export::Subsystem {
        export::Subsystem {
            class: subsystemclassesroot[self.class].clone(),
            visibility: self.visibility,
            health: self.health,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipAI {
    pub id: usize,
    pub nav_threshold: f32,
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<export::UnitClassID, f32>,
    pub resource_attract: HashMap<usize, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
    pub friendly_supply_attract: f32,
    pub hostile_supply_attract: f32,
    pub allegiance_demand_attract: f32,
    pub enemy_demand_attract: f32,
    pub strategic_weapon_damage_attract: f32,
    pub strategic_weapon_engine_damage_attract: f32,
    pub strategic_weapon_subsystem_damage_attract: f32,
    pub strategic_weapon_healing_attract: f32,
    pub strategic_weapon_engine_healing_attract: f32,
    pub strategic_weapon_subsystem_healing_attract: f32,
}

impl ShipAI {
    fn desiccate(self_entity: &export::ShipAI) -> ShipAI {
        ShipAI {
            id: self_entity.id,
            nav_threshold: self_entity.nav_threshold,
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
            strategic_weapon_damage_attract: self_entity.strategic_weapon_damage_attract,
            strategic_weapon_engine_damage_attract: self_entity
                .strategic_weapon_engine_damage_attract,
            strategic_weapon_subsystem_damage_attract: self_entity
                .strategic_weapon_subsystem_damage_attract,
            strategic_weapon_healing_attract: self_entity.strategic_weapon_healing_attract,
            strategic_weapon_engine_healing_attract: self_entity
                .strategic_weapon_engine_healing_attract,
            strategic_weapon_subsystem_healing_attract: self_entity
                .strategic_weapon_subsystem_healing_attract,
        }
    }
    fn rehydrate(&self, resourcesroot: &Vec<Arc<export::Resource>>) -> export::ShipAI {
        export::ShipAI {
            id: self.id,
            nav_threshold: self.nav_threshold,
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
            strategic_weapon_damage_attract: self.strategic_weapon_damage_attract,
            strategic_weapon_engine_damage_attract: self.strategic_weapon_engine_damage_attract,
            strategic_weapon_subsystem_damage_attract: self
                .strategic_weapon_subsystem_damage_attract,
            strategic_weapon_healing_attract: self.strategic_weapon_healing_attract,
            strategic_weapon_engine_healing_attract: self.strategic_weapon_engine_healing_attract,
            strategic_weapon_subsystem_healing_attract: self
                .strategic_weapon_subsystem_healing_attract,
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
    fn desiccate(self_entity: &export::UnitLocation) -> UnitLocation {
        match self_entity {
            export::UnitLocation::Node(n) => UnitLocation::Node(n.id),
            export::UnitLocation::Squadron(s) => UnitLocation::Squadron(s.id),
            export::UnitLocation::Hangar(h) => UnitLocation::Hangar(h.id),
        }
    }
    fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
        hangarslist: &Vec<Arc<export::Hangar>>,
    ) -> export::UnitLocation {
        match self {
            UnitLocation::Node(n) => export::UnitLocation::Node(nodesroot[*n].clone()),
            UnitLocation::Squadron(s) => export::UnitLocation::Squadron(
                squadronsroot
                    .iter()
                    .find(|squadron| &squadron.id == s)
                    .unwrap()
                    .clone(),
            ),
            UnitLocation::Hangar(h) => export::UnitLocation::Hangar(
                hangarslist
                    .iter()
                    .find(|hangar| &hangar.id == h)
                    .unwrap()
                    .clone(),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub shipflavor: usize,
    pub base_hull: u64,     //how many hull hitpoints this ship has by default
    pub base_strength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub propagates: bool,
    pub hangarvol: u64,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub default_weapons: Option<HashMap<usize, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<usize>,
    pub engines: Vec<usize>,
    pub repairers: Vec<usize>,
    pub strategicweapons: Vec<usize>,
    pub factories: Vec<usize>,
    pub shipyards: Vec<usize>,
    pub subsystems: Vec<usize>,
    pub ai_class: usize,
    pub processor_demand_nav_scalar: f32, //multiplier for demand generated by the ship's engines, repairers, factories, and shipyards, to modify it relative to that generated by stockpiles
    pub mother_loyalty_scalar: f32,
    pub deploys_self: bool, //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments; value is number of moves a daughter must be able to make to be deployed
    pub mother_misalignment_tolerance: Option<f32>,
    pub defectchance: HashMap<usize, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub toughness_scalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub battle_escape_scalar: f32, //is added to toughness_scalar in battles where this ship is on the losing side, trying to escape
    pub defect_escape_scalar: f32, //influences how likely it is that a ship of this class will, if it defects, escape to an enemy-held node with no engagement taking place
    pub interdiction_scalar: f32,
    pub strategic_weapon_evasion_scalar: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this shipclass to be
}

impl ShipClass {
    fn desiccate(self_entity: &export::ShipClass) -> ShipClass {
        ShipClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            shipflavor: self_entity.shipflavor.id,
            base_hull: self_entity.base_hull,
            base_strength: self_entity.base_strength,
            visibility: self_entity.visibility,
            propagates: self_entity.propagates,
            hangarvol: self_entity.hangar_vol,
            stockpiles: self_entity
                .stockpiles
                .iter()
                .map(|x| PluripotentStockpile::desiccate(x))
                .collect(),
            default_weapons: self_entity.default_weapons.clone().map(|x| {
                x.iter()
                    .map(|(resource, count)| (resource.id, *count))
                    .collect()
            }),
            hangars: self_entity.hangars.iter().map(|x| x.id).collect(),
            engines: self_entity.engines.iter().map(|x| x.id).collect(),
            repairers: self_entity.repairers.iter().map(|x| x.id).collect(),
            strategicweapons: self_entity.strategic_weapons.iter().map(|x| x.id).collect(),
            factories: self_entity.factories.iter().map(|x| x.id).collect(),
            shipyards: self_entity.shipyards.iter().map(|x| x.id).collect(),
            subsystems: self_entity.subsystems.iter().map(|x| x.id).collect(),
            ai_class: self_entity.ai_class.id,
            processor_demand_nav_scalar: self_entity.processor_demand_nav_scalar,
            deploys_self: self_entity.deploys_self,
            deploys_daughters: self_entity.deploys_daughters,
            mother_loyalty_scalar: self_entity.mother_loyalty_scalar,
            mother_misalignment_tolerance: self_entity.mother_misalignment_tolerance,
            defectchance: self_entity
                .defect_chance
                .iter()
                .map(|(faction, scalars)| (faction.id, *scalars))
                .collect(),
            toughness_scalar: self_entity.toughness_scalar,
            battle_escape_scalar: self_entity.battle_escape_scalar,
            defect_escape_scalar: self_entity.defect_escape_scalar,
            interdiction_scalar: self_entity.interdiction_scalar,
            strategic_weapon_evasion_scalar: self_entity.strategic_weapon_evasion_scalar,
            value_mult: self_entity.value_mult,
        }
    }
    fn rehydrate(
        &self,
        factionsroot: &Vec<Arc<export::Faction>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
        hangarclassesroot: &Vec<Arc<export::HangarClass>>,
        engineclassesroot: &Vec<Arc<export::EngineClass>>,
        repairerclassesroot: &Vec<Arc<export::RepairerClass>>,
        strategicweaponclassesroot: &Vec<Arc<export::StrategicWeaponClass>>,
        factoryclassesroot: &Vec<Arc<export::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<export::ShipyardClass>>,
        subsystemclassesroot: &Vec<Arc<export::SubsystemClass>>,
        shipaisroot: &Vec<Arc<export::ShipAI>>,
        shipflavorsroot: &Vec<Arc<export::ShipFlavor>>,
    ) -> export::ShipClass {
        export::ShipClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            shipflavor: shipflavorsroot[self.shipflavor].clone(),
            base_hull: self.base_hull,
            base_strength: self.base_strength,
            visibility: self.visibility,
            propagates: self.propagates,
            hangar_vol: self.hangarvol,
            stockpiles: self
                .stockpiles
                .iter()
                .map(|x| x.rehydrate(&resourcesroot))
                .collect(),
            default_weapons: self.default_weapons.clone().map(|x| {
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
            strategic_weapons: self
                .strategicweapons
                .iter()
                .map(|x| strategicweaponclassesroot[*x].clone())
                .collect(),
            factories: self
                .factories
                .iter()
                .map(|x| factoryclassesroot[*x].clone())
                .collect(),
            shipyards: self
                .shipyards
                .iter()
                .map(|x| shipyardclassesroot[*x].clone())
                .collect(),
            subsystems: self
                .subsystems
                .iter()
                .map(|x| subsystemclassesroot[*x].clone())
                .collect(),
            ai_class: shipaisroot[self.ai_class].clone(),
            processor_demand_nav_scalar: self.processor_demand_nav_scalar,
            deploys_self: self.deploys_self,
            deploys_daughters: self.deploys_daughters,
            mother_loyalty_scalar: self.mother_loyalty_scalar,
            mother_misalignment_tolerance: self.mother_misalignment_tolerance,
            defect_chance: self
                .defectchance
                .iter()
                .map(|(faction, scalars)| (factionsroot[*faction].clone(), *scalars))
                .collect(),
            toughness_scalar: self.toughness_scalar,
            battle_escape_scalar: self.battle_escape_scalar,
            defect_escape_scalar: self.defect_escape_scalar,
            interdiction_scalar: self.interdiction_scalar,
            strategic_weapon_evasion_scalar: self.strategic_weapon_evasion_scalar,
            value_mult: self.value_mult,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipMut {
    pub hull: export::ShipHealth, //how many hitpoints the ship has
    pub visibility: bool,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<Hangar>,
    pub engines: Vec<Engine>,
    pub movement_left: u64, //starts at eighteen quintillion each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<Repairer>,
    pub strategicweapons: Vec<StrategicWeapon>,
    pub factories: Vec<Factory>,
    pub shipyards: Vec<Shipyard>,
    pub subsystems: Vec<Subsystem>,
    pub location: UnitLocation, //where the ship is -- a node if it's unaffiliated, a squadron if it's in one
    pub allegiance: usize,      //which faction this ship belongs to
    pub last_mother: Option<u64>,
    pub objectives: Vec<Objective>,
    pub aiclass: usize,
}

impl ShipMut {
    fn desiccate(self_entity: &export::ShipMut) -> ShipMut {
        ShipMut {
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
                .map(|x| Hangar::desiccate(x))
                .collect(),
            engines: self_entity
                .engines
                .iter()
                .map(|x| Engine::desiccate(x))
                .collect(),
            movement_left: self_entity.movement_left,
            repairers: self_entity
                .repairers
                .iter()
                .map(|x| Repairer::desiccate(x))
                .collect(),
            strategicweapons: self_entity
                .strategic_weapons
                .iter()
                .map(|x| StrategicWeapon::desiccate(x))
                .collect(),
            factories: self_entity
                .factories
                .iter()
                .map(|x| Factory::desiccate(x))
                .collect(),
            shipyards: self_entity
                .shipyards
                .iter()
                .map(|x| Shipyard::desiccate(x))
                .collect(),
            subsystems: self_entity
                .subsystems
                .iter()
                .map(|x| Subsystem::desiccate(x))
                .collect(),
            location: UnitLocation::desiccate(&self_entity.location),
            allegiance: self_entity.allegiance.id,
            last_mother: self_entity.last_mother,
            objectives: self_entity
                .objectives
                .iter()
                .map(|x| Objective::desiccate(x))
                .collect(),
            aiclass: self_entity.ai_class.id,
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        nodesroot: &Vec<Arc<export::Node>>,
        edgeflavorsroot: &Vec<Arc<export::EdgeFlavor>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
        engineclassesroot: &Vec<Arc<export::EngineClass>>,
        repairerclassesroot: &Vec<Arc<export::RepairerClass>>,
        strategicweaponclassesroot: &Vec<Arc<export::StrategicWeaponClass>>,
        factoryclassesroot: &Vec<Arc<export::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<export::ShipyardClass>>,
        subsystemclassesroot: &Vec<Arc<export::SubsystemClass>>,
        shipaisroot: &Vec<Arc<export::ShipAI>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
    ) -> export::ShipMut {
        export::ShipMut {
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
            strategic_weapons: self
                .strategicweapons
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &strategicweaponclassesroot))
                .collect(),
            factories: self
                .factories
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &factoryclassesroot))
                .collect(),
            shipyards: self
                .shipyards
                .iter()
                .map(|x| x.rehydrate(&resourcesroot, &shipyardclassesroot, &shipclassesroot))
                .collect(),
            subsystems: self
                .subsystems
                .iter()
                .map(|x| x.rehydrate(&subsystemclassesroot))
                .collect(),
            location: export::UnitLocation::Node(nodesroot[0].clone()),
            allegiance: factionsroot[self.allegiance].clone(),
            last_mother: self.last_mother,
            objectives: Vec::new(),
            ai_class: shipaisroot[self.aiclass].clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ship {
    pub id: u64,
    pub visible_name: String,
    pub class: usize, //which class of ship this is
    pub mutables: ShipMut,
}

impl Ship {
    fn desiccate(self_entity: &export::Ship) -> Ship {
        Ship {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            class: self_entity.class.id,
            mutables: ShipMut::desiccate(&self_entity.mutables.read().unwrap()),
        }
    }
    fn rehydrate(
        &self,
        nodeflavorsroot: &Vec<Arc<export::NodeFlavor>>,
        nodesroot: &Vec<Arc<export::Node>>,
        edgeflavorsroot: &Vec<Arc<export::EdgeFlavor>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        resourcesroot: &Vec<Arc<export::Resource>>,
        engineclassesroot: &Vec<Arc<export::EngineClass>>,
        repairerclassesroot: &Vec<Arc<export::RepairerClass>>,
        strategicweaponclassesroot: &Vec<Arc<export::StrategicWeaponClass>>,
        factoryclassesroot: &Vec<Arc<export::FactoryClass>>,
        shipyardclassesroot: &Vec<Arc<export::ShipyardClass>>,
        subsystemclassesroot: &Vec<Arc<export::SubsystemClass>>,
        shipaisroot: &Vec<Arc<export::ShipAI>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
    ) -> export::Ship {
        export::Ship {
            id: self.id,
            visible_name: self.visible_name.clone(),
            class: shipclassesroot[self.class].clone(),
            mutables: RwLock::new(self.mutables.rehydrate(
                nodeflavorsroot,
                nodesroot,
                edgeflavorsroot,
                factionsroot,
                resourcesroot,
                engineclassesroot,
                repairerclassesroot,
                strategicweaponclassesroot,
                factoryclassesroot,
                shipyardclassesroot,
                subsystemclassesroot,
                shipaisroot,
                shipclassesroot,
            )),
        }
    }
    fn add_hangars_and_objectives(
        ship: &Arc<export::Ship>,
        connectionships: &Vec<Ship>,
        nodesroot: &Vec<Arc<export::Node>>,
        clustersroot: &Vec<Arc<export::Cluster>>,
        hangarclassesroot: &Vec<Arc<export::HangarClass>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> Vec<Arc<export::Hangar>> {
        let connection_ship = connectionships
            .iter()
            .find(|connectionship| &connectionship.id == &ship.id)
            .unwrap();
        let root_hangars: Vec<Arc<export::Hangar>> = connection_ship
            .mutables
            .hangars
            .iter()
            .map(|x| Arc::new(x.rehydrate(&hangarclassesroot, &shipsroot, &squadronsroot)))
            .collect();
        ship.mutables.write().unwrap().hangars = root_hangars.clone();
        ship.mutables.write().unwrap().objectives = connection_ship
            .mutables
            .objectives
            .iter()
            .map(|x| x.rehydrate(&nodesroot, &clustersroot, &shipsroot, &squadronsroot))
            .collect();
        root_hangars
    }
    fn set_location(
        ship: &Arc<export::Ship>,
        connectionships: &Vec<Ship>,
        nodesroot: &Vec<Arc<export::Node>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
        hangarslist: &Vec<Arc<export::Hangar>>,
    ) {
        ship.mutables.write().unwrap().location = connectionships
            .iter()
            .find(|connectionship| &connectionship.id == &ship.id)
            .unwrap()
            .mutables
            .location
            .rehydrate(&nodesroot, &squadronsroot, &hangarslist);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub squadronflavor: usize,
    pub visibility: bool,
    pub capacity: u64,
    pub target: u64,
    pub propagates: bool,
    pub strength_mod: (f32, u64),
    pub allowed: Option<Vec<export::UnitClassID>>,
    pub ideal: HashMap<export::UnitClassID, u64>,
    pub sub_target_supply_scalar: f32, //multiplier used for demand generated by non-ideal ships under the target limit; should be below one
    pub non_ideal_demand_scalar: f32, //multiplier used for demand generated for non-ideal unitclasses; should be below one
    pub nav_quorum: f32,
    pub creation_threshold: f32,
    pub de_ghost_threshold: f32,
    pub disband_threshold: f32,
    pub deploys_self: bool, //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments
    pub mother_loyalty_scalar: f32,
    pub defect_chance: HashMap<usize, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub defect_escape_mod: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this squadronclass to be
}

impl SquadronClass {
    fn desiccate(self_entity: &export::SquadronClass) -> SquadronClass {
        SquadronClass {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            description: self_entity.description.clone(),
            squadronflavor: self_entity.squadronflavor.id,
            visibility: self_entity.visibility,
            target: self_entity.target,
            capacity: self_entity.capacity,
            propagates: self_entity.propagates,
            strength_mod: self_entity.strength_mod.clone(),
            allowed: self_entity.allowed.clone(),
            ideal: self_entity.ideal.clone(),
            sub_target_supply_scalar: self_entity.sub_target_supply_scalar,
            non_ideal_demand_scalar: self_entity.non_ideal_demand_scalar,
            nav_quorum: self_entity.nav_quorum,
            creation_threshold: self_entity.creation_threshold,
            de_ghost_threshold: self_entity.de_ghost_threshold,
            disband_threshold: self_entity.disband_threshold,
            deploys_self: self_entity.deploys_self,
            deploys_daughters: self_entity.deploys_daughters,
            mother_loyalty_scalar: self_entity.mother_loyalty_scalar,
            defect_chance: self_entity
                .defect_chance
                .iter()
                .map(|(faction, scalars)| (faction.id, *scalars))
                .collect(),
            defect_escape_mod: self_entity.defect_escape_mod.clone(),
            value_mult: self_entity.value_mult.clone(),
        }
    }
    fn rehydrate(
        &self,
        factionsroot: &Vec<Arc<export::Faction>>,
        squadronflavorsroot: &Vec<Arc<export::SquadronFlavor>>,
    ) -> export::SquadronClass {
        export::SquadronClass {
            id: self.id,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            squadronflavor: squadronflavorsroot[self.squadronflavor].clone(),
            visibility: self.visibility,
            target: self.target,
            capacity: self.capacity,
            propagates: self.propagates,
            strength_mod: self.strength_mod.clone(),
            allowed: self.allowed.clone(),
            ideal: self.ideal.clone(),
            sub_target_supply_scalar: self.sub_target_supply_scalar,
            non_ideal_demand_scalar: self.non_ideal_demand_scalar,
            nav_quorum: self.nav_quorum,
            creation_threshold: self.creation_threshold,
            de_ghost_threshold: self.de_ghost_threshold,
            disband_threshold: self.disband_threshold,
            deploys_self: self.deploys_self,
            deploys_daughters: self.deploys_daughters,
            mother_loyalty_scalar: self.mother_loyalty_scalar,
            defect_chance: self
                .defect_chance
                .iter()
                .map(|(faction, scalars)| (factionsroot[*faction].clone(), *scalars))
                .collect(),
            defect_escape_mod: self.defect_escape_mod.clone(),
            value_mult: self.value_mult.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronMut {
    pub visibility: bool,
    pub location: UnitLocation,
    pub allegiance: usize,
    pub last_mother: Option<u64>,
    pub objectives: Vec<Objective>,
    pub ghost: bool,
}

impl SquadronMut {
    fn desiccate(self_entity: &export::SquadronMut) -> SquadronMut {
        SquadronMut {
            visibility: self_entity.visibility,
            location: UnitLocation::desiccate(&self_entity.location),
            allegiance: self_entity.allegiance.id,
            last_mother: self_entity.last_mother,
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
        nodesroot: &Vec<Arc<export::Node>>,
        factionsroot: &Vec<Arc<export::Faction>>,
    ) -> export::SquadronMut {
        export::SquadronMut {
            visibility: self.visibility,
            location: export::UnitLocation::Node(nodesroot[0].clone()),
            allegiance: factionsroot[self.allegiance].clone(),
            last_mother: self.last_mother,
            objectives: Vec::new(),
            ghost: self.ghost,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Squadron {
    pub id: u64,
    pub visible_name: String,
    pub class: usize,
    pub idealstrength: u64,
    pub mutables: SquadronMut,
    pub unit_container: UnitContainer,
}

impl Squadron {
    fn desiccate(self_entity: &export::Squadron) -> Squadron {
        Squadron {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            class: self_entity.class.id,
            idealstrength: self_entity.ideal_strength,
            mutables: SquadronMut::desiccate(&self_entity.mutables.read().unwrap()),
            unit_container: UnitContainer::desiccate(&self_entity.unit_container.read().unwrap()),
        }
    }
    fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        squadronclassesroot: &Vec<Arc<export::SquadronClass>>,
    ) -> export::Squadron {
        export::Squadron {
            id: self.id,
            visible_name: self.visible_name.clone(),
            class: squadronclassesroot[self.class].clone(),
            ideal_strength: self.idealstrength,
            mutables: RwLock::new(self.mutables.rehydrate(&nodesroot, &factionsroot)),
            unit_container: RwLock::new(export::UnitContainer::new()),
        }
    }
    fn add_daughters_and_objectives_set_location(
        squadron: &Arc<export::Squadron>,
        connectionsquadrons: &Vec<Squadron>,
        nodesroot: &Vec<Arc<export::Node>>,
        clustersroot: &Vec<Arc<export::Cluster>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
        hangarslist: &Vec<Arc<export::Hangar>>,
    ) {
        let connection_squadron = connectionsquadrons
            .iter()
            .find(|c_s| c_s.id == squadron.id)
            .unwrap();
        squadron.unit_container.write().unwrap().contents = connection_squadron
            .unit_container
            .contents
            .iter()
            .map(|x| x.rehydrate(&shipsroot, &squadronsroot))
            .collect();
        squadron.mutables.write().unwrap().objectives = connection_squadron
            .mutables
            .objectives
            .iter()
            .map(|x| x.rehydrate(&nodesroot, &clustersroot, &shipsroot, &squadronsroot))
            .collect();
        squadron.mutables.write().unwrap().location = connection_squadron
            .mutables
            .location
            .rehydrate(&nodesroot, &squadronsroot, &hangarslist);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum UnitClass {
    ShipClass(usize),
    SquadronClass(usize),
}

impl UnitClass {
    fn desiccate(self_entity: &export::UnitClass) -> UnitClass {
        match self_entity {
            export::UnitClass::ShipClass(shc) => UnitClass::ShipClass(shc.id),
            export::UnitClass::SquadronClass(sqc) => UnitClass::SquadronClass(sqc.id),
        }
    }
    fn rehydrate(
        &self,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
        squadronclassesroot: &Vec<Arc<export::SquadronClass>>,
    ) -> export::UnitClass {
        match self {
            UnitClass::ShipClass(shc) => {
                export::UnitClass::ShipClass(shipclassesroot[*shc].clone())
            }
            UnitClass::SquadronClass(sqc) => {
                export::UnitClass::SquadronClass(squadronclassesroot[*sqc].clone())
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
    fn desiccate(self_entity: &export::Unit) -> Unit {
        match self_entity {
            export::Unit::Ship(sh) => Unit::Ship(sh.id),
            export::Unit::Squadron(sq) => Unit::Squadron(sq.id),
        }
    }
    fn rehydrate(
        &self,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> export::Unit {
        match self {
            Unit::Ship(sh) => export::Unit::Ship(
                shipsroot
                    .iter()
                    .find(|ship| &ship.id == sh)
                    .unwrap()
                    .clone(),
            ),
            Unit::Squadron(sq) => export::Unit::Squadron(
                squadronsroot
                    .iter()
                    .find(|squadron| &squadron.id == sq)
                    .unwrap()
                    .clone(),
            ),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct UnitRecord {
    id: u64,
    visible_name: String,
    class: UnitClass,
    allegiance: usize,
    daughters: Vec<u64>,
}

impl UnitRecord {
    fn desiccate(self_entity: &export::UnitRecord) -> UnitRecord {
        UnitRecord {
            id: self_entity.id,
            visible_name: self_entity.visible_name.clone(),
            class: UnitClass::desiccate(&self_entity.class),
            allegiance: self_entity.allegiance.id,
            daughters: self_entity.daughters.clone(),
        }
    }
    fn rehydrate(
        &self,
        factionsroot: &Vec<Arc<export::Faction>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
        squadronclassesroot: &Vec<Arc<export::SquadronClass>>,
    ) -> export::UnitRecord {
        export::UnitRecord {
            id: self.id,
            visible_name: self.visible_name.clone(),
            class: self.class.rehydrate(shipclassesroot, squadronclassesroot),
            allegiance: factionsroot[self.allegiance].clone(),
            daughters: self.daughters.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveTarget {
    Node(usize),
    Cluster(usize),
    Unit(Unit),
}

impl ObjectiveTarget {
    pub fn desiccate(self_entity: &export::ObjectiveTarget) -> ObjectiveTarget {
        match self_entity {
            export::ObjectiveTarget::Node(node) => ObjectiveTarget::Node(node.id),
            export::ObjectiveTarget::Cluster(cluster) => ObjectiveTarget::Cluster(cluster.id),
            export::ObjectiveTarget::Unit(unit) => ObjectiveTarget::Unit(Unit::desiccate(unit)),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        clustersroot: &Vec<Arc<export::Cluster>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> export::ObjectiveTarget {
        match self {
            ObjectiveTarget::Node(node) => export::ObjectiveTarget::Node(nodesroot[*node].clone()),
            ObjectiveTarget::Cluster(cluster) => {
                export::ObjectiveTarget::Cluster(clustersroot[*cluster].clone())
            }
            ObjectiveTarget::Unit(unit) => {
                export::ObjectiveTarget::Unit(unit.rehydrate(shipsroot, squadronsroot))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub target: ObjectiveTarget,
    pub task: export::ObjectiveTask,
    pub fraction: Option<f32>,
    pub duration: Option<u64>,
    pub time_limit: Option<u64>,
    pub difficulty: f32,
    pub cost: u64,
    pub durationscalar: f32,
    pub strengthscalar: f32,
    pub toughness_scalar: f32,
    pub battleescapescalar: f32,
}

impl Objective {
    pub fn desiccate(self_entity: &export::Objective) -> Objective {
        Objective {
            target: ObjectiveTarget::desiccate(&self_entity.target),
            task: self_entity.task,
            fraction: self_entity.fraction,
            duration: self_entity.duration,
            time_limit: self_entity.time_limit,
            difficulty: self_entity.difficulty,
            cost: self_entity.cost,
            durationscalar: self_entity.duration_scalar,
            strengthscalar: self_entity.strength_scalar,
            toughness_scalar: self_entity.toughness_scalar,
            battleescapescalar: self_entity.battle_escape_scalar,
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        clustersroot: &Vec<Arc<export::Cluster>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> export::Objective {
        export::Objective {
            target: self
                .target
                .rehydrate(nodesroot, clustersroot, shipsroot, squadronsroot),
            task: self.task,
            fraction: self.fraction,
            duration: self.duration,
            time_limit: self.time_limit,
            difficulty: self.difficulty,
            cost: self.cost,
            duration_scalar: self.durationscalar,
            strength_scalar: self.strengthscalar,
            toughness_scalar: self.toughness_scalar,
            battle_escape_scalar: self.battleescapescalar,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    pub visible_name: String,
    pub objectives: Vec<Objective>,
}

impl Operation {
    pub fn desiccate(self_entity: &export::Operation) -> Operation {
        Operation {
            visible_name: self_entity.visible_name.clone(),
            objectives: self_entity
                .objectives
                .iter()
                .map(|x| Objective::desiccate(x))
                .collect(),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        clustersroot: &Vec<Arc<export::Cluster>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
    ) -> export::Operation {
        export::Operation {
            visible_name: self.visible_name.clone(),
            objectives: self
                .objectives
                .iter()
                .map(|x| x.rehydrate(&nodesroot, &clustersroot, &shipsroot, &squadronsroot))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct FactionForcesRecord {
    pub local_forces: HashMap<UnitRecord, UnitStatus>,
    pub reinforcements: Vec<(usize, u64, HashMap<UnitRecord, UnitStatus>)>,
}

impl FactionForcesRecord {
    pub fn desiccate(self_entity: &export::FactionForcesRecord) -> FactionForcesRecord {
        FactionForcesRecord {
            local_forces: self_entity
                .local_forces
                .iter()
                .map(|(record, status)| {
                    (UnitRecord::desiccate(record), UnitStatus::desiccate(status))
                })
                .collect(),
            reinforcements: self_entity
                .reinforcements
                .iter()
                .map(|(node, distance, units)| {
                    (
                        node.id,
                        *distance,
                        units
                            .iter()
                            .map(|(record, status)| {
                                (UnitRecord::desiccate(record), UnitStatus::desiccate(status))
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
        squadronclassesroot: &Vec<Arc<export::SquadronClass>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
        hangarslist: &Vec<Arc<export::Hangar>>,
    ) -> export::FactionForcesRecord {
        export::FactionForcesRecord {
            local_forces: self
                .local_forces
                .iter()
                .map(|(record, status)| {
                    (
                        record.rehydrate(factionsroot, shipclassesroot, squadronclassesroot),
                        status.rehydrate(nodesroot, squadronsroot, hangarslist),
                    )
                })
                .collect(),
            reinforcements: self
                .reinforcements
                .iter()
                .map(|(node, distance, units)| {
                    (
                        nodesroot[*node].clone(),
                        *distance,
                        units
                            .iter()
                            .map(|(record, status)| {
                                (
                                    record.rehydrate(
                                        factionsroot,
                                        shipclassesroot,
                                        squadronclassesroot,
                                    ),
                                    status.rehydrate(nodesroot, squadronsroot, hangarslist),
                                )
                            })
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
    pub damage: i64,
    pub engine_damage: Vec<i64>,
    pub strategic_weapon_damage: Vec<i64>,
}

impl UnitStatus {
    pub fn desiccate(self_entity: &export::UnitStatus) -> UnitStatus {
        UnitStatus {
            location: self_entity
                .location
                .clone()
                .map(|x| UnitLocation::desiccate(&x)),
            damage: self_entity.damage,
            engine_damage: self_entity.engine_damage.clone(),
            strategic_weapon_damage: self_entity.subsystem_damage.clone(),
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
        hangarslist: &Vec<Arc<export::Hangar>>,
    ) -> export::UnitStatus {
        export::UnitStatus {
            location: self
                .location
                .clone()
                .map(|x| x.rehydrate(&nodesroot, &squadronsroot, &hangarslist)),
            damage: self.damage,
            engine_damage: self.engine_damage.clone(),
            subsystem_damage: self.strategic_weapon_damage.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngagementRecord {
    pub visible_name: String,
    pub turn: u64,
    pub coalitions: HashMap<u64, HashMap<usize, FactionForcesRecord>>,
    pub aggressor: Option<usize>,
    pub objectives: HashMap<usize, Vec<Objective>>,
    pub location: usize,
    pub duration: u64,
    pub victors: (usize, u64),
}

impl EngagementRecord {
    pub fn desiccate(self_entity: &export::EngagementRecord) -> EngagementRecord {
        EngagementRecord {
            visible_name: self_entity.visible_name.clone(),
            turn: self_entity.turn,
            coalitions: self_entity
                .coalitions
                .iter()
                .map(|(index, faction_map)| {
                    (
                        *index,
                        faction_map
                            .iter()
                            .map(|(faction, forces)| {
                                (faction.id, FactionForcesRecord::desiccate(forces))
                            })
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
        }
    }
    pub fn rehydrate(
        &self,
        nodesroot: &Vec<Arc<export::Node>>,
        clustersroot: &Vec<Arc<export::Cluster>>,
        factionsroot: &Vec<Arc<export::Faction>>,
        shipclassesroot: &Vec<Arc<export::ShipClass>>,
        squadronclassesroot: &Vec<Arc<export::SquadronClass>>,
        shipsroot: &Vec<Arc<export::Ship>>,
        squadronsroot: &Vec<Arc<export::Squadron>>,
        hangarslist: &Vec<Arc<export::Hangar>>,
    ) -> export::EngagementRecord {
        export::EngagementRecord {
            visible_name: self.visible_name.clone(),
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
                                    forces.rehydrate(
                                        &nodesroot,
                                        &factionsroot,
                                        &shipclassesroot,
                                        &squadronclassesroot,
                                        &shipsroot,
                                        &squadronsroot,
                                        &hangarslist,
                                    ),
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
                                x.rehydrate(&nodesroot, &clustersroot, &shipsroot, &squadronsroot)
                            })
                            .collect(),
                    )
                })
                .collect(),
            location: nodesroot[self.location].clone(),
            duration: self.duration,
            victors: (factionsroot[self.victors.0].clone(), self.victors.1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalSalience {
    pub faction_salience: Vec<Vec<Vec<[f32; 2]>>>,
    pub resource_salience: Vec<Vec<Vec<[f32; 2]>>>,
    pub unitclass_salience: Vec<Vec<Vec<[f32; 2]>>>,
    pub strategic_weapon_effect_map: Vec<Vec<[(i64, f32); 3]>>,
}

impl GlobalSalience {
    fn desiccate(self_entity: &export::GlobalSalience) -> GlobalSalience {
        GlobalSalience {
            faction_salience: self_entity.faction_salience.read().unwrap().clone(),
            resource_salience: self_entity.resource_salience.read().unwrap().clone(),
            unitclass_salience: self_entity.unitclass_salience.read().unwrap().clone(),
            strategic_weapon_effect_map: self_entity
                .strategic_weapon_effect_map
                .read()
                .unwrap()
                .clone(),
        }
    }
    fn rehydrate(&self) -> export::GlobalSalience {
        export::GlobalSalience {
            faction_salience: RwLock::new(self.faction_salience.clone()),
            resource_salience: RwLock::new(self.resource_salience.clone()),
            unitclass_salience: RwLock::new(self.unitclass_salience.clone()),
            strategic_weapon_effect_map: RwLock::new(self.strategic_weapon_effect_map.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Root {
    pub config: export::Config,
    pub nodeflavors: Vec<export::NodeFlavor>,
    pub nodes: Vec<Node>,
    pub clusters: Vec<Cluster>,
    pub edgeflavors: Vec<export::EdgeFlavor>,
    pub edges: HashMap<(usize, usize), usize>,
    pub neighbors: HashMap<usize, Vec<usize>>,
    pub factions: Vec<export::Faction>,
    pub wars: HashSet<(usize, usize)>,
    pub resources: Vec<export::Resource>,
    pub hangarclasses: Vec<export::HangarClass>,
    pub hangarcounter: u64,
    pub engineclasses: Vec<EngineClass>,
    pub repairerclasses: Vec<RepairerClass>,
    pub strategicweaponclasses: Vec<StrategicWeaponClass>,
    pub factoryclasses: Vec<FactoryClass>,
    pub shipyardclasses: Vec<ShipyardClass>,
    pub subsystemclasses: Vec<export::SubsystemClass>,
    pub shipais: Vec<ShipAI>,
    pub shipflavors: Vec<export::ShipFlavor>,
    pub squadronflavors: Vec<export::SquadronFlavor>,
    pub shipclasses: Vec<ShipClass>,
    pub squadronclasses: Vec<SquadronClass>,
    pub ships: Vec<Ship>,
    pub squadrons: Vec<Squadron>,
    pub unitcounter: u64,
    pub engagements: Vec<EngagementRecord>,
    pub globalsalience: GlobalSalience,
    pub turn: u64,
}

impl Root {
    pub fn desiccate(self_entity: &export::Root) -> Root {
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
            clusters: self_entity
                .clusters
                .iter()
                .map(|x| Cluster::desiccate(x))
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
            hangarcounter: self_entity.hangar_counter.load(atomic::Ordering::Relaxed),
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
            strategicweaponclasses: self_entity
                .strategicweaponclasses
                .iter()
                .map(|x| StrategicWeaponClass::desiccate(x))
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
            subsystemclasses: self_entity
                .subsystemclasses
                .iter()
                .map(|x| Arc::unwrap_or_clone(x.clone()))
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
            ships: self_entity
                .ships
                .read()
                .unwrap()
                .iter()
                .map(|x| Ship::desiccate(x))
                .collect(),
            squadrons: self_entity
                .squadrons
                .read()
                .unwrap()
                .iter()
                .map(|x| Squadron::desiccate(x))
                .collect(),
            unitcounter: self_entity.unit_counter.load(atomic::Ordering::Relaxed),
            engagements: self_entity
                .engagements
                .read()
                .unwrap()
                .iter()
                .map(|x| EngagementRecord::desiccate(x))
                .collect(),
            globalsalience: GlobalSalience::desiccate(&self_entity.global_salience),
            turn: self_entity.turn.load(atomic::Ordering::Relaxed),
        }
    }
    pub fn rehydrate(mut self) -> export::Root {
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
        let hangarcounter = Arc::new(AtomicU64::new(self.hangarcounter));
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
        let subsystemclasses = self
            .subsystemclasses
            .drain(0..)
            .map(|x| Arc::new(x))
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
        let strategicweaponclasses = self
            .strategicweaponclasses
            .drain(0..)
            .map(|x| Arc::new(x.rehydrate(&resources, &nodeflavors, &edgeflavors, &shipflavors)))
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
                    &strategicweaponclasses,
                    &factoryclasses,
                    &shipyardclasses,
                    &subsystemclasses,
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
        let clusters = self
            .clusters
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
        let ships: RwLock<Vec<Arc<export::Ship>>> = RwLock::new(
            self.ships
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
                        &strategicweaponclasses,
                        &factoryclasses,
                        &shipyardclasses,
                        &subsystemclasses,
                        &shipais,
                        &shipclasses,
                    ))
                })
                .collect(),
        );
        let squadrons = RwLock::new(
            self.squadrons
                .iter()
                .map(|x| Arc::new(x.rehydrate(&nodes, &factions, &squadronclasses)))
                .collect(),
        );
        //here we go through and add the units to the nodes, now that we have the data for unit s
        nodes.iter().for_each(|node| {
            Node::add_units(
                node,
                &self.nodes,
                &ships.read().unwrap(),
                &squadrons.read().unwrap(),
            )
        });
        let hangarslist = ships
            .read()
            .unwrap()
            .iter()
            .map(|ship| {
                Ship::add_hangars_and_objectives(
                    ship,
                    &self.ships,
                    &nodes,
                    &clusters,
                    &hangarclasses,
                    &ships.read().unwrap(),
                    &squadrons.read().unwrap(),
                )
            })
            .flatten()
            .collect();
        ships.read().unwrap().iter().for_each(|ship| {
            Ship::set_location(
                ship,
                &self.ships,
                &nodes,
                &squadrons.read().unwrap(),
                &hangarslist,
            )
        });
        squadrons.read().unwrap().iter().for_each(|squadron| {
            Squadron::add_daughters_and_objectives_set_location(
                squadron,
                &self.squadrons,
                &nodes,
                &clusters,
                &ships.read().unwrap(),
                &squadrons.read().unwrap(),
                &hangarslist,
            )
        });
        let unitcounter = Arc::new(AtomicU64::new(self.unitcounter));
        let engagements = RwLock::new(
            self.engagements
                .drain(0..)
                .map(|x| {
                    x.rehydrate(
                        &nodes,
                        &clusters,
                        &factions,
                        &shipclasses,
                        &squadronclasses,
                        &ships.read().unwrap(),
                        &squadrons.read().unwrap(),
                        &hangarslist,
                    )
                })
                .collect(),
        );
        let globalsalience = self.globalsalience.rehydrate();
        let turn = Arc::new(AtomicU64::new(self.turn));

        export::Root {
            config,
            nodeflavors,
            nodes,
            clusters,
            edgeflavors,
            edges,
            neighbors,
            factions,
            wars,
            resources,
            hangarclasses,
            hangar_counter: hangarcounter,
            engineclasses,
            repairerclasses,
            strategicweaponclasses,
            factoryclasses,
            shipyardclasses,
            subsystemclasses,
            shipais,
            shipflavors,
            squadronflavors,
            shipclasses,
            squadronclasses,
            ships,
            squadrons,
            unit_counter: unitcounter,
            engagements,
            global_salience: globalsalience,
            turn,
        }
    }
}
