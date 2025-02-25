use crate::internal::engagement::{EngagementPrep, UnitRecord};
use crate::internal::faction::Faction;
use crate::internal::hangar::{Hangar, HangarClass, UnitContainer};
use crate::internal::node::{Locality, Node};
use crate::internal::resource::{
    Engine, EngineClass, Factory, FactoryClass, FactoryState, PluripotentStockpile, Repairer,
    RepairerClass, Resource, ResourceProcess, Shipyard, ShipyardClass, Stockpileness,
    StrategicWeapon, StrategicWeaponClass,
};
use crate::internal::root::{Objective, Root};
use crate::internal::salience::transpose;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_distr::*;
use rand_hc::Hc128Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::iter;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubsystemClass {
    pub id: usize,
    pub visible_name: String,
    pub visibility: bool,
    pub base_health: Option<u64>,
    pub toughness_scalar: f32,
    pub strength_mod: (f32, u64),
}

impl SubsystemClass {
    fn instantiate(class: Arc<Self>) -> Subsystem {
        Subsystem {
            class: class.clone(),
            visibility: class.visibility,
            health: class.base_health,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Subsystem {
    pub class: Arc<SubsystemClass>,
    pub visibility: bool,
    pub health: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ShipAI {
    pub id: usize,
    pub nav_threshold: f32,
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<UnitClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
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

impl PartialEq for ShipAI {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipAI {}

impl Ord for ShipAI {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipAI {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipAI {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone)]
pub enum UnitLocation {
    Node(Arc<Node>),
    Squadron(Arc<Squadron>),
    Hangar(Arc<Hangar>),
}

impl UnitLocation {
    pub fn get_unit_container_write(&self) -> RwLockWriteGuard<UnitContainer> {
        match self {
            UnitLocation::Node(node) => node.unit_container.write().unwrap(),
            UnitLocation::Squadron(squadron) => squadron.unit_container.write().unwrap(),
            UnitLocation::Hangar(hangar) => hangar.unit_container.write().unwrap(),
        }
    }
    pub fn get_ship_mut_read(&self) -> Option<RwLockReadGuard<ShipMut>> {
        match self {
            UnitLocation::Node(_) => None,
            UnitLocation::Squadron(_) => None,
            UnitLocation::Hangar(hangar) => Some(hangar.mother.mutables.read().unwrap()),
        }
    }
    pub fn is_alive_locked(
        &self,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
    ) -> bool {
        match self {
            UnitLocation::Node(_) => true,
            UnitLocation::Squadron(squadron) => {
                squadrons_mut_lock.get(&squadron.id).unwrap().ghost
                    || *container_is_not_empty_map.get(&squadron.id).unwrap()
            }
            UnitLocation::Hangar(hangar) => {
                ships_mut_lock.get(&hangar.mother.id).unwrap().hull.get() > 0
            }
        }
    }
    pub fn check_insert(&self, container: &RwLockWriteGuard<UnitContainer>, unit: Unit) -> bool {
        match self {
            UnitLocation::Node(_node) => true,
            UnitLocation::Squadron(squadron) => {
                (squadron
                    .class
                    .allowed
                    .as_ref()
                    .map(|allowed_vec| {
                        allowed_vec
                            .contains(&UnitClassID::new_from_unitclass(&unit.get_unitclass()))
                    })
                    .unwrap_or(true))
                    && unit.get_capacity_volume()
                        <= squadron.class.capacity
                            - container
                                .contents
                                .iter()
                                .map(|unit| unit.get_capacity_volume())
                                .sum::<u64>()
            }
            UnitLocation::Hangar(hangar) => {
                (hangar
                    .class
                    .allowed
                    .as_ref()
                    .map(|allowed_vec| {
                        allowed_vec
                            .contains(&UnitClassID::new_from_unitclass(&unit.get_unitclass()))
                    })
                    .unwrap_or(true))
                    && (unit.get_capacity_volume()
                        <= hangar.class.capacity
                            - container
                                .contents
                                .iter()
                                .map(|unit| unit.get_capacity_volume())
                                .sum::<u64>())
            }
        }
    }
    pub fn check_remove(&self, container: &RwLockWriteGuard<UnitContainer>, unit: Unit) -> bool {
        container.contents.contains(&unit)
    }
    pub fn insert_unit(&self, container: &mut RwLockWriteGuard<UnitContainer>, unit: Unit) {
        container.contents.push(unit.clone());
    }
    pub fn remove_unit(&self, container: &mut RwLockWriteGuard<UnitContainer>, unit: Unit) {
        container.contents.retain(|content| content != &unit);
    }
    pub fn get_mother_node(&self) -> Arc<Node> {
        match self {
            UnitLocation::Node(node) => node.clone(),
            UnitLocation::Squadron(squadron) => squadron.get_mother_node(),
            UnitLocation::Hangar(hangar) => hangar.mother.get_mother_node(),
        }
    }
}

impl PartialEq for UnitLocation {
    fn eq(&self, other: &Self) -> bool {
        let self_val = match self {
            UnitLocation::Node(n) => n.id,
            UnitLocation::Squadron(s) => s.id as usize,
            UnitLocation::Hangar(h) => h.id as usize,
        };
        let other_val = match other {
            UnitLocation::Node(n) => n.id,
            UnitLocation::Squadron(s) => s.id as usize,
            UnitLocation::Hangar(h) => h.id as usize,
        };
        self_val == other_val
    }
}

pub trait Mobility {
    fn get_unit(&self) -> Unit;
    fn get_unitclass(&self) -> UnitClass;
    fn get_ship(&self) -> Option<Arc<Ship>>;
    fn get_squadron(&self) -> Option<Arc<Squadron>>;
    fn get_id(&self) -> u64;
    fn get_visible_name(&self) -> String;
    fn is_ship(&self) -> bool;
    fn get_location(&self) -> UnitLocation;
    fn check_location_coherency(&self);
    fn is_in_node(&self) -> bool {
        match self.get_location() {
            UnitLocation::Node(_) => true,
            _ => false,
        }
    }
    fn is_in_squadron(&self) -> bool {
        match self.get_location() {
            UnitLocation::Squadron(_) => true,
            _ => false,
        }
    }
    fn is_in_hangar(&self) -> bool {
        match self.get_location() {
            UnitLocation::Hangar(_) => true,
            _ => false,
        }
    }
    fn get_mother_node(&self) -> Arc<Node> {
        match self.get_location() {
            UnitLocation::Node(node) => node,
            UnitLocation::Squadron(squadron) => squadron.get_mother_node(),
            UnitLocation::Hangar(hangar) => hangar.mother.get_mother_node(),
        }
    }
    fn get_mother_unit(&self) -> Option<Unit> {
        match self.get_location() {
            UnitLocation::Node(_) => None,
            UnitLocation::Squadron(squadron) => Some(squadron.get_unit()),
            UnitLocation::Hangar(hangar) => Some(hangar.mother.get_unit()),
        }
    }
    fn get_base_hull(&self) -> u64;
    fn get_hull(&self) -> u64;
    fn get_engine_base_health(&self) -> u64;
    fn get_subsystem_base_health(&self) -> u64;
    fn get_allegiance(&self) -> Arc<Faction>;
    fn get_daughters(&self) -> Vec<Unit>;
    fn get_daughters_recursive(&self) -> Vec<Unit>;
    fn get_undocked_daughters(&self) -> Vec<Arc<Ship>>;
    fn get_mother_loyalty_scalar(&self) -> f32;
    fn get_morale_scalar(&self) -> f32;
    fn get_character_strength_scalar(&self) -> f32;
    fn get_interdiction_scalar(&self) -> f32;
    fn get_processor_demand_nav_scalar(&self) -> f32;
    fn get_strength(&self, time: u64) -> u64;
    fn get_strength_locked(
        &self,
        hangars_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
        time: u64,
    ) -> u64;
    fn get_strength_post_engagement(&self, damage: i64) -> u64;
    fn get_real_volume(&self) -> u64;
    fn get_real_volume_locked(
        &self,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
    ) -> u64;
    fn get_capacity_volume(&self) -> u64;
    fn get_ai(&self) -> NavAI;
    fn get_nav_threshold(&self) -> f32;
    fn get_objectives(&self) -> Vec<Objective>;
    fn get_deployment_threshold(&self) -> Option<u64>;
    fn get_deployment_status(&self) -> bool;
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32);
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64;
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64;
    fn get_unitclass_supply_recursive(&self, unitclass: UnitClass) -> u64;
    fn get_unitclass_demand_recursive(&self, unitclass: UnitClass) -> u64;
    fn change_allegiance(&self, new_faction: Arc<Faction>);
    fn acyclicity_check(&self, location: UnitLocation) -> bool {
        match location.clone() {
            UnitLocation::Squadron(squadron) => {
                if squadron.get_id() == self.get_id() {
                    false
                } else {
                    self.acyclicity_check(squadron.get_location())
                }
            }
            UnitLocation::Hangar(hangar) => {
                let carrier = hangar.mother.clone();
                if carrier.get_id() == self.get_id() {
                    false
                } else {
                    self.acyclicity_check(carrier.get_location())
                }
            }
            _ => true,
        }
    }
    fn acyclicity_check_locked(
        &self,
        location: UnitLocation,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
    ) -> bool {
        match location.clone() {
            UnitLocation::Squadron(squadron) => {
                let squadron_id = squadron.get_id();
                let squadron_mut = squadrons_mut_lock.get(&squadron_id).unwrap();
                if squadron_id == self.get_id() {
                    false
                } else {
                    self.acyclicity_check_locked(
                        squadron_mut.location.clone(),
                        ships_mut_lock,
                        squadrons_mut_lock,
                    )
                }
            }
            UnitLocation::Hangar(hangar) => {
                let carrier = hangar.mother.clone();
                let carrier_id = carrier.get_id();
                let carrier_mut = ships_mut_lock.get(&carrier_id).unwrap();
                if carrier_id == self.get_id() {
                    false
                } else {
                    self.acyclicity_check_locked(
                        carrier_mut.location.clone(),
                        ships_mut_lock,
                        squadrons_mut_lock,
                    )
                }
            }
            _ => true,
        }
    }
    fn transfer(&self, destination: UnitLocation) -> bool;
    fn transfer_locked(
        &self,
        source: UnitLocation,
        source_container: &mut RwLockWriteGuard<UnitContainer>,
        destination: UnitLocation,
        destination_container: &mut RwLockWriteGuard<UnitContainer>,
        ships_mut_lock: &mut HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &mut HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
    ) -> bool {
        if self.is_alive_locked(
            ships_mut_lock,
            squadrons_mut_lock,
            container_is_not_empty_map,
        ) && source.is_alive_locked(
            ships_mut_lock,
            squadrons_mut_lock,
            container_is_not_empty_map,
        ) && destination.is_alive_locked(
            ships_mut_lock,
            squadrons_mut_lock,
            container_is_not_empty_map,
        ) && source.check_remove(&source_container, self.get_unit())
            && destination.check_insert(&destination_container, self.get_unit())
            && self.acyclicity_check_locked(destination.clone(), ships_mut_lock, squadrons_mut_lock)
        {
            source.remove_unit(source_container, self.get_unit());
            match self.get_ship() {
                Some(ship) => {
                    let ship_mut = ships_mut_lock.get_mut(&ship.id).unwrap();
                    ship_mut.location = destination.clone();
                    match destination.clone() {
                        UnitLocation::Hangar(hangar) => {
                            ship_mut.last_mother = Some(hangar.mother.id)
                        }
                        UnitLocation::Squadron(squadron) => {
                            ship_mut.last_mother = Some(squadron.id)
                        }
                        _ => {}
                    }
                }
                None => {
                    let squadron_mut = squadrons_mut_lock.get_mut(&self.get_id()).unwrap();
                    squadron_mut.location = destination.clone();
                    match destination.clone() {
                        UnitLocation::Hangar(hangar) => {
                            squadron_mut.last_mother = Some(hangar.mother.id)
                        }
                        UnitLocation::Squadron(squadron) => {
                            squadron_mut.last_mother = Some(squadron.id)
                        }
                        _ => {}
                    }
                }
            };
            destination.insert_unit(destination_container, self.get_unit());
            true
        } else {
            false
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>>;
    //The ship and squadron implementations of this next method do slightly different things.
    //The ship version is used for gathering reinforcements, and assumes the ship can't make the move and its daughters will have to move independently.
    //It recurses down the tree, following that logic at every stage.
    //The squadron version is used to determine how much of the squadron can make a particular move.
    //Since granddaughters won't leave their mothers to accompany a squadron that leaves them behind, this version just checks the immediate daughters.
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit>;
    fn set_movement_recursive(&self, value: u64);
    fn get_moves_left(&self, turn: u64) -> u64;
    fn process_engines(&self, root: &Root, destination: Arc<Node>);
    fn get_node_nav_attractiveness(&self, root: &Root, node: Arc<Node>) -> NotNan<f32> {
        let location: Arc<Node> = self.get_mother_node();
        let allegiance = self.get_allegiance();
        let ai = self.get_ai();

        let resource_salience = &root.global_salience.resource_salience.read().unwrap();

        //this checks how much value the node holds with regards to resources the subject ship is seeking
        let resource_demand_value: f32 = ai
            .resource_attract
            .iter()
            .map(|(resource, scalar)| {
                let demand = resource_salience[allegiance.id][resource.id][node.id][0];
                let supply = resource_salience[allegiance.id][resource.id][node.id][1];
                (demand - supply) * self.get_resource_supply(resource.clone()) as f32 * scalar
            })
            .sum();
        let resource_supply_value: f32 = ai
            .resource_attract
            .iter()
            .map(|(resource, scalar)| {
                //we index into the salience map by resource and then by node
                //to determine how much supply there is in this node for each resource the subject ship wants
                //NOTE: Previously, we got demand by indexing by nodeid, not position.
                //I believe using the ship's current position to calculate demand
                //will eliminate a pathology and produce more correct gradient-following behavior.
                let demand = resource_salience[allegiance.id][resource.id][node.id][0];
                let supply = resource_salience[allegiance.id][resource.id][node.id][1];
                ((((supply * demand) + ((supply - demand) * 5.0)) / 10.0)
                    * self.get_resource_demand_from_stockpiles(resource.clone()) as f32
                    * scalar)
                    + (supply
                        * self.get_resource_demand_from_processors(resource.clone()) as f32
                        * self.get_processor_demand_nav_scalar())
            })
            .sum();

        let unitclass_salience = &root.global_salience.unitclass_salience.read().unwrap();

        let unitclass_demand_value: f32 = ai
            .ship_cargo_attract
            .iter()
            .map(|(unitclassid, scalar)| {
                let attractive_unitclass = match unitclassid {
                    UnitClassID::ShipClass(shipclassid) => UnitClass::ShipClass(
                        root.shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                    UnitClassID::SquadronClass(squadronclassid) => UnitClass::SquadronClass(
                        root.squadronclasses
                            .iter()
                            .find(|squadronclass| squadronclass.id == squadronclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                };
                let demand = unitclass_salience[allegiance.id][unitclassid.get_index()][node.id][0];
                let supply = unitclass_salience[allegiance.id][unitclassid.get_index()][node.id][1];
                (demand - supply)
                    * self.get_unitclass_supply_recursive(attractive_unitclass.clone()) as f32
                    * scalar
            })
            .sum();
        //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
        let unitclass_supply_value: f32 = ai
            .ship_cargo_attract
            .iter()
            .map(|(unitclassid, scalar)| {
                //we index into the salience map by resource and then by node
                //to determine how much supply there is in this node for each resource the subject ship wants
                //NOTE: Previously, we got demand by indexing by nodeid, not location.
                //I believe using the ship's current position to calculate demand
                //will eliminate a pathology and produce more correct gradient-following behavior.
                let attractive_unitclass = match unitclassid {
                    UnitClassID::ShipClass(shipclassid) => UnitClass::ShipClass(
                        root.shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                    UnitClassID::SquadronClass(squadronclassid) => UnitClass::SquadronClass(
                        root.squadronclasses
                            .iter()
                            .find(|squadronclass| squadronclass.id == squadronclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                };
                let demand =
                    unitclass_salience[allegiance.id][unitclassid.get_index()][location.id][0];
                let supply = unitclass_salience[allegiance.id][unitclassid.get_index()][node.id][1];
                (((supply * demand) + ((supply - demand) * 5.0)) / 10.0)
                    * self.get_unitclass_demand_recursive(attractive_unitclass.clone()) as f32
                    * scalar
            })
            .sum();
        //this checks how much demand there is in the node for ships of the subject ship's class
        let ship_value_specific: f32 = unitclass_salience[allegiance.id]
            [self.get_unitclass().get_id()][node.id][0]
            * ai.ship_attract_specific;
        //oh, THIS is why we needed the placeholder ship class
        //this checks how much demand there is in the node for ships in general
        let ship_value_generic: f32 =
            unitclass_salience[allegiance.id][0][node.id][0] * ai.ship_attract_generic;

        let faction_salience = &root.global_salience.faction_salience.read().unwrap();

        let faction_supply = transpose(&faction_salience[allegiance.id])[node.id]
            .iter()
            .map(|array| array[0])
            .sum::<f32>();
        let friendly_supply_value =
            faction_supply.clamp(0.0, f32::MAX) * ai.friendly_supply_attract;
        let hostile_supply_value = faction_supply.clamp(f32::MIN, 0.0) * ai.hostile_supply_attract;

        let allegiance_supply_unscaled = faction_salience[allegiance.id][allegiance.id][node.id][0];
        let allegiance_demand_value = (faction_salience[allegiance.id][allegiance.id][node.id][1]
            - allegiance_supply_unscaled)
            .clamp(0.0, f32::MAX)
            * ai.allegiance_demand_attract;

        let enemy_supply_unscaled = root
            .factions
            .iter()
            .filter(|faction| {
                root.wars.contains(&(
                    faction.clone().min(&allegiance.clone().clone()).clone(),
                    allegiance.clone().max(faction.clone().clone()).clone(),
                ))
            })
            .map(|faction| faction_salience[faction.id][faction.id][node.id][0])
            .sum::<f32>();
        let enemy_demand_unscaled = root
            .factions
            .iter()
            .filter(|faction| {
                root.wars.contains(&(
                    faction.clone().min(&allegiance.clone().clone()).clone(),
                    allegiance.clone().max(faction.clone().clone()).clone(),
                ))
            })
            .map(|faction| faction_salience[faction.id][faction.id][node.id][1])
            .sum::<f32>();
        let enemy_demand_value = (enemy_demand_unscaled - enemy_supply_unscaled)
            .clamp(0.0, f32::MAX)
            * ai.allegiance_demand_attract;

        let weapon_effect = root
            .global_salience
            .strategic_weapon_effect_map
            .read()
            .unwrap()[allegiance.id][node.id];

        let strategic_weapon_damage_value =
            (weapon_effect[0].0.clamp(0, self.get_base_hull() as i64) as f32
                + (weapon_effect[0].1.clamp(0.0, 1.0) * self.get_base_hull() as f32))
                * ai.strategic_weapon_damage_attract;

        let strategic_weapon_engine_damage_value = (weapon_effect[1]
            .0
            .clamp(0, self.get_engine_base_health() as i64)
            as f32
            + (weapon_effect[1].1.clamp(0.0, 1.0) * self.get_engine_base_health() as f32))
            * ai.strategic_weapon_engine_damage_attract;

        let strategic_weapon_subsystem_damage_value = (weapon_effect[2]
            .0
            .clamp(0, self.get_subsystem_base_health() as i64)
            as f32
            + (weapon_effect[2].1.clamp(0.0, 1.0) * self.get_subsystem_base_health() as f32))
            * ai.strategic_weapon_subsystem_damage_attract;

        let strategic_weapon_healing_value =
            (weapon_effect[0].0.clamp(-(self.get_base_hull() as i64), 0) as f32
                + (weapon_effect[0].1.clamp(-1.0, 0.0) * self.get_base_hull() as f32))
                * ai.strategic_weapon_healing_attract;

        let strategic_weapon_engine_healing_value = (weapon_effect[1]
            .0
            .clamp(-(self.get_engine_base_health() as i64), 0)
            as f32
            + (weapon_effect[1].1.clamp(-1.0, 0.0) * self.get_engine_base_health() as f32))
            * ai.strategic_weapon_engine_healing_attract;

        let strategic_weapon_subsystem_healing_value = (weapon_effect[2]
            .0
            .clamp(-(self.get_subsystem_base_health() as i64), 0)
            as f32
            + (weapon_effect[2].1.clamp(-1.0, 0.0) * self.get_subsystem_base_health() as f32))
            * ai.strategic_weapon_subsystem_healing_attract;

        NotNan::new(
            resource_demand_value
                + resource_supply_value
                + unitclass_demand_value
                + unitclass_supply_value
                + ship_value_specific
                + ship_value_generic
                + friendly_supply_value
                + hostile_supply_value
                + allegiance_demand_value
                + enemy_demand_value
                + strategic_weapon_damage_value
                + strategic_weapon_engine_damage_value
                + strategic_weapon_subsystem_damage_value
                + strategic_weapon_healing_value
                + strategic_weapon_engine_healing_value
                + strategic_weapon_subsystem_healing_value,
        )
        .unwrap()
    }
    fn navigate(
        //used for ships which are operating independently
        //this method determines which of the current node's neighbors is most desirable
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Arc<Node>> {
        let location = self.get_mother_node();
        //we iterate over the destinations to determine which neighbor is most desirable
        let neighbor_values: Vec<(Arc<Node>, NotNan<f32>)> = destinations
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .map(|node| {
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                (
                    node.clone(),
                    self.get_node_nav_attractiveness(root, node.clone()),
                )
            })
            .collect();
        let null_pair = (location.clone(), NotNan::new(0.0).unwrap());
        let best_neighbor = neighbor_values
            .iter()
            .max_by_key(|(_, val)| val)
            .unwrap_or(&null_pair);
        if best_neighbor.1
            > self.get_node_nav_attractiveness(root, location) * self.get_nav_threshold()
        {
            Some(best_neighbor.0.clone())
        } else {
            None
        }
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>>;
    fn maneuver(&self, root: &Root) -> Option<Arc<Node>> {
        let neighbors = root
            .neighbors
            .get(&self.get_mother_node())
            .unwrap_or(&Vec::new())
            .clone();
        if (self.is_alive()) && self.is_in_node() {
            if let Some(destinations) = self.destinations_check(root, &neighbors) {
                let destination_option = self.navigate(root, &destinations);
                match destination_option.clone() {
                    Some(destination) => {
                        self.traverse(root, destination.clone());
                        if let Some(aggressor) =
                            root.engagement_check(destination.clone(), self.get_allegiance())
                        {
                            let engagement = EngagementPrep::engagement_prep(
                                root,
                                destination.clone(),
                                Some(aggressor),
                            )
                            .internal_battle(root);
                            engagement.battle_cleanup(root);
                        }
                        self.deploy_daughters(root);
                        if self.get_daughters_recursive().iter().any(|daughter| {
                            daughter
                                .get_ship()
                                .map(|ship| {
                                    ship.mutables.read().unwrap().strategic_weapons.len() != 0
                                })
                                .unwrap_or(false)
                        }) {
                            //NOTE: Check whether this will actually assign properly.
                            *root
                                .global_salience
                                .strategic_weapon_effect_map
                                .write()
                                .unwrap() = root.calculate_strategic_weapon_effect_map();
                        }
                        destination.transact_resources(root);
                        destination.transact_units(root);
                    }
                    None => {}
                }
                destination_option
            } else {
                None
            }
        } else {
            None
        }
    }
    fn deploy_daughters(&self, root: &Root) {
        if let Some(threshold) = self.get_deployment_threshold() {
            let daughters = self.get_daughters();
            let (active_daughters, passive_daughters): (Vec<&Unit>, Vec<&Unit>) =
                daughters.iter().partition(|daughter| {
                    daughter.get_deployment_status()
                        && (daughter.get_moves_left(root.turn.load(atomic::Ordering::Relaxed))
                            > threshold)
                });
            active_daughters.iter().for_each(|daughter| {
                daughter.transfer(UnitLocation::Node(self.get_mother_node()));
                let mut moving = true;
                while moving {
                    if daughter.maneuver(root).is_none() {
                        moving = false
                    }
                }
            });
            passive_daughters
                .iter()
                .for_each(|daughter| daughter.deploy_daughters(root));
        }
    }
    fn calculate_damage(
        &self,
        root: &Root,
        is_victor: bool,
        victor_strength: f32,
        victis_strength: f32,
        duration: u64,
        duration_damage_rand: f32,
        rng: &mut Hc128Rng,
    ) -> (i64, Vec<i64>, Vec<i64>);
    fn damage(&self, damage: i64, engine_damage: &Vec<i64>, strategic_weapon_damage: &Vec<i64>);
    fn repair(&self, per_engagement: bool);
    //Checks whether the unit will defect this turn; if it will, makes the unit defect and returns the node the ship is in afterward
    //so that that node can be checked for engagements
    fn try_defect(&self, root: &Root) -> Option<Vec<Arc<Node>>> {
        let location = self.get_mother_node();
        let allegiance = self.get_allegiance();
        let (defectchance, defectescapescalar) = self.get_defection_data();
        //NOTE: Test whether this gives us numbers that make sense.
        let local_threat_ratio: f32 = defectchance
            .iter()
            .map(|(faction, _)| {
                root.global_salience.faction_salience.read().unwrap()[allegiance.id][faction.id]
                    [location.id][0]
            })
            .sum();
        //if defectchance only contains data for one faction
        //then either it's the faction it's currently part of, in which case we have no data on its chance of joining other factions
        //or it's not, in which case we have no data on its chance of defecting from its current faction
        //so either way we treat it as incapable of defecting
        let defect_probability = if defectchance.len() > 1 {
            ((local_threat_ratio * defectchance
                .get(&allegiance)
                .unwrap_or(&(0.0, 0.0))
                .0)
                //NOTE: I *think* that clamp will resolve things properly if we end up dividing by zero -- it'll set the probability to 1 -- but it's hard to be sure
                /self.get_morale_scalar())
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut rng = Hc128Rng::seed_from_u64(47);
        let defects = rng.gen_bool(defect_probability as f64);
        if defects {
            let interdiction: f32 = location
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .filter(|unit| unit.get_allegiance() == allegiance)
                .filter(|unit| unit.get_id() != self.get_id())
                .map(|unit| unit.get_interdiction_scalar())
                .product();
            let new_faction_probabilities: Vec<(Arc<Faction>, f32)> = defectchance
                .iter()
                .map(|(faction, (_, defect_to))| {
                    (
                        faction.clone(),
                        (defect_to
                            * root.global_salience.faction_salience.read().unwrap()[faction.id]
                                [faction.id][location.id][0]),
                    )
                })
                .collect();
            let new_faction: Arc<Faction> = new_faction_probabilities
                .choose_weighted(&mut rng, |(_, prob)| prob.clone())
                .unwrap()
                .0
                .clone();
            self.change_allegiance(new_faction.clone());
            //NOTE: This should take more things into account probably
            let escapes = rng.gen_bool((defectescapescalar / interdiction).clamp(0.0, 1.0) as f64);
            if escapes {
                let destinations_option =
                    self.destinations_check(root, root.neighbors.get(&location).unwrap());
                match destinations_option.clone() {
                    Some(destinations) => {
                        let destination = destinations
                            .iter()
                            .max_by_key(|node| {
                                root.global_salience.faction_salience.read().unwrap()
                                    [new_faction.id][new_faction.id][node.id][0]
                                    as i64
                            })
                            .unwrap()
                            .clone();
                        self.traverse(root, destination.clone());
                        Some(vec![location, destination])
                    }
                    None => {
                        self.transfer(UnitLocation::Node(location.clone()));
                        Some(vec![location])
                    }
                }
            } else {
                self.transfer(UnitLocation::Node(location.clone()));
                Some(vec![location])
            }
        } else {
            None
        }
    }
    fn kill(&self);
    fn is_alive(&self) -> bool;
    fn is_alive_locked(
        &self,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
    ) -> bool;
    fn record(&self) -> UnitRecord;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipFlavor {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
}

impl PartialEq for ShipFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipFlavor {}

impl Ord for ShipFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ShipClassID {
    pub index: usize,
}

impl ShipClassID {
    pub fn new_from_index(index: usize) -> Self {
        ShipClassID { index: index }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShipClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub shipflavor: Arc<ShipFlavor>,
    pub base_hull: u64,     //how many hull hitpoints this ship has by default
    pub base_strength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub propagates: bool,
    pub hangar_vol: u64,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub default_weapons: Option<HashMap<Arc<Resource>, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<Arc<HangarClass>>,
    pub engines: Vec<Arc<EngineClass>>,
    pub repairers: Vec<Arc<RepairerClass>>,
    pub strategic_weapons: Vec<Arc<StrategicWeaponClass>>,
    pub factories: Vec<Arc<FactoryClass>>,
    pub shipyards: Vec<Arc<ShipyardClass>>,
    pub subsystems: Vec<Arc<SubsystemClass>>,
    pub ai_class: Arc<ShipAI>,
    pub processor_demand_nav_scalar: f32, //multiplier for demand generated by the ship's processors, to modify it relative to that generated by stockpiles used for transport
    pub deploys_self: bool,               //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments; value is number of moves a daughter must be able to make to be deployed
    pub mother_loyalty_scalar: f32,
    pub mother_misalignment_tolerance: Option<f32>, //if the ratio between this ship's AI and its mother's AI exceeds this value, the ship will leave its mother
    pub defect_chance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub toughness_scalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub battle_escape_scalar: f32, //is added to toughness_scalar in battles where this ship is on the losing side, trying to escape
    pub defect_escape_scalar: f32, //influences how likely it is that a ship of this class will, if it defects, escape to an enemy-held node with no engagement taking place
    pub interdiction_scalar: f32,
    pub strategic_weapon_evasion_scalar: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this shipclass to be
}

impl ShipClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.base_strength
            + self
                .hangars
                .iter()
                .map(|hangarclass| hangarclass.get_ideal_strength(root))
                .sum::<u64>()
    }
    pub fn get_ideal_volume(&self) -> u64 {
        self.hangar_vol
    }
    pub fn get_unitclass(class: Arc<Self>) -> UnitClass {
        UnitClass::ShipClass(class.clone())
    }
    //method to create a ship  with this ship class
    //to avoid an infinite loop, this does not give the ship any hangars
    //we generate those in a later step with build_hangars
    pub fn instantiate(
        class: Arc<Self>,
        location: UnitLocation,
        faction: Arc<Faction>,
        root: &Root,
    ) -> Ship {
        let index = root.unit_counter.fetch_add(1, atomic::Ordering::Relaxed);
        Ship {
            id: index,
            visible_name: uuid::Uuid::new_v4().to_string(),
            class: class.clone(),
            mutables: RwLock::new(ShipMut {
                hull: ShipHealth::new(class.base_hull),
                visibility: class.visibility,
                stockpiles: class.stockpiles.clone(),
                efficiency: 1.0,
                hangars: Vec::new(),
                engines: class
                    .engines
                    .iter()
                    .map(|engineclass| EngineClass::instantiate(engineclass.clone()))
                    .collect(),
                movement_left: u64::MAX,
                repairers: class
                    .repairers
                    .iter()
                    .map(|repairerclass| RepairerClass::instantiate(repairerclass.clone()))
                    .collect(),
                strategic_weapons: class
                    .strategic_weapons
                    .iter()
                    .map(|strategicweaponclass| {
                        StrategicWeaponClass::instantiate(strategicweaponclass.clone())
                    })
                    .collect(),
                factories: class
                    .factories
                    .iter()
                    .map(|factoryclass| FactoryClass::instantiate(factoryclass.clone()))
                    .collect(),
                shipyards: class
                    .shipyards
                    .iter()
                    .map(|shipyardclass| {
                        ShipyardClass::instantiate(shipyardclass.clone(), &root.shipclasses)
                    })
                    .collect(),
                subsystems: class
                    .subsystems
                    .iter()
                    .map(|subsystemclass| SubsystemClass::instantiate(subsystemclass.clone()))
                    .collect(),
                location,
                allegiance: faction,
                last_mother: None,
                objectives: Vec::new(),
                ai_class: class.ai_class.clone(),
            }),
        }
    }
    //NOTE: having this be a method feels a little odd when instantiate isn't one
    pub fn build_hangars(
        &self,
        ship: Arc<Ship>,
        shipclasses: &Vec<Arc<ShipClass>>,
        counter: &Arc<AtomicU64>,
    ) {
        let hangars: Vec<_> = self
            .hangars
            .iter()
            .map(|hangarclass| {
                Arc::new(HangarClass::instantiate(
                    hangarclass.clone(),
                    ship.clone(),
                    shipclasses,
                    counter,
                ))
            })
            .collect();
        ship.mutables.write().unwrap().hangars = hangars;
    }
}

impl Eq for ShipClass {}

impl Ord for ShipClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Eq, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct ShipHealth {
    health: u64,
}

impl ShipHealth {
    pub fn new(val: u64) -> Self {
        ShipHealth { health: val }
    }
    pub fn get(&self) -> u64 {
        self.health
    }
    pub fn set(&mut self, val: u64) {
        if self.health > 0 {
            self.health = val
        }
    }
    pub fn add(&mut self, val: u64) {
        if self.health > 0 {
            self.health += val
        }
    }
    pub fn sub(&mut self, val: u64) {
        self.health = self.health.saturating_sub(val)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ShipMut {
    pub hull: ShipHealth, //how many hitpoints the ship has
    pub visibility: bool,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<Arc<Hangar>>,
    pub engines: Vec<Engine>,
    pub movement_left: u64, //starts at u64::MAX each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<Repairer>,
    pub strategic_weapons: Vec<StrategicWeapon>,
    pub factories: Vec<Factory>,
    pub shipyards: Vec<Shipyard>,
    pub subsystems: Vec<Subsystem>,
    pub location: UnitLocation, //where the ship is -- a node if it's unaffiliated, a squadron if it's in one
    pub allegiance: Arc<Faction>, //which faction this ship belongs to
    pub last_mother: Option<u64>,
    pub objectives: Vec<Objective>,
    pub ai_class: Arc<ShipAI>,
}

impl ShipMut {
    pub fn get_speed_factor(&self, average_speed: u64, turn: u64) -> f32 {
        self.engines
            .iter()
            .find(|engine| engine.check_engine_movement_only(turn))
            .map(|engine| {
                (engine.class.speed as f32 / engine.class.cooldown as f32) / average_speed as f32
            })
            .unwrap_or(0.0)
    }
}

#[derive(Debug)]
pub struct Ship {
    pub id: u64,
    pub visible_name: String,
    pub class: Arc<ShipClass>, //which class of ship this is
    pub mutables: RwLock<ShipMut>,
}

impl Ship {
    pub fn process_factories(&self) {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        mutables
            .factories
            .iter_mut()
            .for_each(|f| f.process(efficiency));
    }
    pub fn process_shipyards(&self) {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        mutables
            .shipyards
            .iter_mut()
            .for_each(|sy| sy.process(efficiency));
    }
    pub fn plan_ships(
        &self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)> {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        let location = mutables.location.clone();
        let allegiance = mutables.allegiance.clone();
        mutables
            .shipyards
            .iter_mut()
            .map(|shipyard| {
                let ship_plans = shipyard.plan_ships(efficiency, shipclasses);
                //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                ship_plans
                    .iter()
                    .map(|ship_plan| (ship_plan.clone(), location.clone(), allegiance.clone()))
                    // <^>>(
                    .collect::<Vec<_>>()
            })
            //we flatten the collection of vecs corresponding to individual shipyards, because we just want to create all the ships and don't care who created them
            .flatten()
            .collect::<Vec<_>>()
    }
    pub fn reset_movement(&self) {
        self.mutables.write().unwrap().movement_left = u64::MAX;
    }
}

impl PartialEq for Ship {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.visible_name == other.visible_name
            && self.class == other.class
            && self.mutables.read().unwrap().clone() == other.mutables.read().unwrap().clone()
    }
}

impl Eq for Ship {}

impl Ord for Ship {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Ship {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Ship {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Mobility for Arc<Ship> {
    fn get_unit(&self) -> Unit {
        Unit::Ship(self.clone())
    }
    fn get_unitclass(&self) -> UnitClass {
        UnitClass::ShipClass(self.class.clone())
    }
    fn get_ship(&self) -> Option<Arc<Ship>> {
        Some(self.clone())
    }
    fn get_squadron(&self) -> Option<Arc<Squadron>> {
        None
    }
    fn get_id(&self) -> u64 {
        self.id
    }
    fn get_visible_name(&self) -> String {
        self.visible_name.clone()
    }
    fn is_ship(&self) -> bool {
        true
    }
    fn get_location(&self) -> UnitLocation {
        self.mutables.read().unwrap().location.clone()
    }
    fn check_location_coherency(&self) {
        let mother = self.get_location();
        let sisters: Vec<_> = match mother.clone() {
            UnitLocation::Node(node) => node
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Squadron(squadron) => squadron
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Hangar(hangar) => hangar
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
        };
        assert_eq!(
            sisters
                .iter()
                .filter(|sister| sister.get_id() == self.id)
                .count(),
            1
        );
    }
    fn get_base_hull(&self) -> u64 {
        self.class.base_hull
    }
    fn get_hull(&self) -> u64 {
        self.mutables.read().unwrap().hull.get()
    }
    fn get_engine_base_health(&self) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .engines
            .iter()
            .map(|e| e.class.base_health.unwrap_or(0))
            .sum()
    }
    fn get_subsystem_base_health(&self) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .subsystems
            .iter()
            .map(|s| s.class.base_health.unwrap_or(0))
            .sum()
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        self.mutables.read().unwrap().allegiance.clone()
    }
    fn get_daughters(&self) -> Vec<Unit> {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| {
                hangar
                    .unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
    fn get_daughters_recursive(&self) -> Vec<Unit> {
        iter::once(self.clone().get_unit())
            .chain(
                self.mutables
                    .read()
                    .unwrap()
                    .hangars
                    .iter()
                    .map(|hangar| {
                        hangar
                            .unit_container
                            .read()
                            .unwrap()
                            .contents
                            .iter()
                            .filter(|unit| unit.is_alive())
                            .map(|unit| unit.get_daughters_recursive())
                            .collect::<Vec<Vec<Unit>>>()
                    })
                    .flatten()
                    .flatten(),
            )
            .collect()
    }
    fn get_undocked_daughters(&self) -> Vec<Arc<Ship>> {
        vec![self.clone()]
    }
    fn get_mother_loyalty_scalar(&self) -> f32 {
        self.class.mother_loyalty_scalar
    }
    //NOTE: Dummied out until morale system exists.
    fn get_morale_scalar(&self) -> f32 {
        1.0
    }
    //NOTE: Dummied out until characters exist.
    fn get_character_strength_scalar(&self) -> f32 {
        1.0
    }
    fn get_interdiction_scalar(&self) -> f32 {
        self.class.interdiction_scalar
            * self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_interdiction_scalar())
                .product::<f32>()
    }
    fn get_processor_demand_nav_scalar(&self) -> f32 {
        self.class.processor_demand_nav_scalar
    }
    fn get_strength(&self, time: u64) -> u64 {
        let mutables = self.mutables.read().unwrap();
        let daughter_strength = mutables
            .hangars
            .iter()
            .map(|hangar| hangar.get_strength(time))
            .sum::<u64>();
        let subsystem_strength_add: u64 = mutables
            .subsystems
            .iter()
            .filter(|subsystem| subsystem.health.map(|val| val > 0).unwrap_or(true))
            .map(|subsystem| subsystem.class.strength_mod.1)
            .sum();
        let subsystem_strength_mult: f32 = mutables
            .subsystems
            .iter()
            .filter(|subsystem| subsystem.health.map(|val| val > 0).unwrap_or(true))
            .map(|subsystem| subsystem.class.strength_mod.0)
            .product();
        let objective_strength: f32 = mutables
            .objectives
            .iter()
            .map(|objective| objective.strength_scalar)
            .product();
        (self.class.base_strength as f32
            * (mutables.hull.get() as f32 / self.class.base_hull as f32)
            * self.get_character_strength_scalar()
            * subsystem_strength_mult
            * objective_strength) as u64
            + subsystem_strength_add
            + daughter_strength
    }
    fn get_strength_locked(
        &self,
        hangars_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
        time: u64,
    ) -> u64 {
        let mutables = ships_mut_lock.get(&self.id).unwrap();
        let daughter_strength = mutables
            .hangars
            .iter()
            .map(|hangar| {
                hangar.get_strength_locked(
                    hangars_containers_lock,
                    squadrons_containers_lock,
                    ships_mut_lock,
                    squadrons_mut_lock,
                    container_is_not_empty_map,
                    time,
                )
            })
            .sum::<u64>();
        let subsystem_strength_add: u64 = mutables
            .subsystems
            .iter()
            .filter(|subsystem| subsystem.health.map(|val| val > 0).unwrap_or(true))
            .map(|subsystem| subsystem.class.strength_mod.1)
            .sum();
        let subsystem_strength_mult: f32 = mutables
            .subsystems
            .iter()
            .filter(|subsystem| subsystem.health.map(|val| val > 0).unwrap_or(true))
            .map(|subsystem| subsystem.class.strength_mod.0)
            .product();
        let objective_strength: f32 = mutables
            .objectives
            .iter()
            .map(|objective| objective.strength_scalar)
            .product();
        (self.class.base_strength as f32
            * (mutables.hull.get() as f32 / self.class.base_hull as f32)
            * self.get_character_strength_scalar()
            * subsystem_strength_mult
            * objective_strength) as u64
            + subsystem_strength_add
            + daughter_strength
    }
    //this one is used for checking which faction has the most strength left after a battle
    //we know how much damage the ship will take, but it hasn't actually been applied yet
    //also, we don't worry about daughters because we're evaluating all units separately
    fn get_strength_post_engagement(&self, damage: i64) -> u64 {
        let mutables = self.mutables.read().unwrap();
        let objective_strength: f32 = mutables
            .objectives
            .iter()
            .map(|objective| objective.strength_scalar)
            .product();
        (self.class.base_strength as f32
            * ((mutables.hull.get() as i64 - damage).clamp(0, self.class.base_hull as i64) as f32
                / self.class.base_hull as f32)
            * self.get_character_strength_scalar()
            * objective_strength) as u64
    }
    fn get_real_volume(&self) -> u64 {
        self.class.hangar_vol
    }
    fn get_real_volume_locked(
        &self,
        _squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        _ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
    ) -> u64 {
        self.class.hangar_vol
    }
    fn get_capacity_volume(&self) -> u64 {
        self.class.hangar_vol
    }
    fn get_ai(&self) -> NavAI {
        let ai = &self.mutables.read().unwrap().ai_class;
        NavAI {
            nav_threshold: ai.nav_threshold,
            ship_attract_specific: ai.ship_attract_specific,
            ship_attract_generic: ai.ship_attract_generic,
            ship_cargo_attract: ai.ship_cargo_attract.clone(),
            resource_attract: ai.resource_attract.clone(),
            friendly_supply_attract: ai.friendly_supply_attract,
            hostile_supply_attract: ai.hostile_supply_attract,
            allegiance_demand_attract: ai.allegiance_demand_attract,
            enemy_demand_attract: ai.enemy_demand_attract,
            strategic_weapon_damage_attract: ai.strategic_weapon_damage_attract,
            strategic_weapon_engine_damage_attract: ai.strategic_weapon_engine_damage_attract,
            strategic_weapon_subsystem_damage_attract: ai.strategic_weapon_subsystem_damage_attract,
            strategic_weapon_healing_attract: ai.strategic_weapon_healing_attract,
            strategic_weapon_engine_healing_attract: ai.strategic_weapon_engine_healing_attract,
            strategic_weapon_subsystem_healing_attract: ai
                .strategic_weapon_subsystem_healing_attract,
        }
    }
    fn get_nav_threshold(&self) -> f32 {
        self.mutables.read().unwrap().ai_class.nav_threshold
    }
    fn get_objectives(&self) -> Vec<Objective> {
        self.mutables
            .read()
            .unwrap()
            .objectives
            .iter()
            .cloned()
            .collect()
    }
    fn get_deployment_threshold(&self) -> Option<u64> {
        self.class.deploys_daughters
    }
    fn get_deployment_status(&self) -> bool {
        self.class.deploys_self
    }
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32) {
        (
            self.class.defect_chance.clone(),
            self.class.defect_escape_scalar,
        )
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        let stockpile_supply = mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>();
        let factory_supply = mutables
            .factories
            .iter()
            .map(|f| f.get_resource_supply_total(resource.clone()))
            .sum::<u64>();
        let daughter_supply = self
            .get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_supply(resource.clone()))
            .sum::<u64>();
        stockpile_supply + factory_supply + daughter_supply
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
            + mutables
                .engines
                .iter()
                .map(|e| e.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .repairers
                .iter()
                .map(|r| r.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .factories
                .iter()
                .map(|f| f.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .shipyards
                .iter()
                .map(|s| s.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_demand(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_demand_from_stockpiles(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .engines
            .iter()
            .map(|e| e.get_resource_demand_total(resource.clone()))
            .sum::<u64>()
            + mutables
                .repairers
                .iter()
                .map(|r| r.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .factories
                .iter()
                .map(|f| f.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .shipyards
                .iter()
                .map(|s| s.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_demand_from_processors(resource.clone()))
                .sum::<u64>()
    }
    //NOTE: Here in get_unitclass_num/unitclass_supply, we use an approach where we count the ship itself if it matches the subject unitclass.
    //Therefore, we don't need to check how many of an entity's direct daughters match the class alongside feeding the method down to them;
    //we can just give the daughters (or the hangars) the method.
    //This allows for more elegant recursion, but may have disadvantages.
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        (&self.get_unitclass() == &unitclass) as u64
            + self
                .mutables
                .read()
                .unwrap()
                .hangars
                .iter()
                .filter(|hangar| hangar.class.propagates)
                .map(|hangar| hangar.get_unitclass_num_recursive(unitclass.clone()))
                .sum::<u64>()
    }
    //NOTE: It's possible that this approach of gathering the volume data one ship at a time like this is less performant than filtering down the collection of units,
    //getting length, then multiplying by volume
    fn get_unitclass_supply_recursive(&self, unitclass: UnitClass) -> u64 {
        ((&self.get_unitclass() == &unitclass) as u64 * self.get_real_volume())
            + self
                .mutables
                .read()
                .unwrap()
                .hangars
                .iter()
                .filter(|hangar| hangar.class.propagates)
                .map(|hangar| hangar.get_unitclass_supply_recursive(unitclass.clone()))
                .sum::<u64>()
    }
    fn get_unitclass_demand_recursive(&self, unitclass: UnitClass) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .filter(|hangar| hangar.class.propagates)
            .map(|hangar| hangar.get_unitclass_demand_recursive(unitclass.clone()))
            .sum::<u64>()
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        self.mutables.write().unwrap().allegiance = new_faction.clone();
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.change_allegiance(new_faction.clone()));
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        let source = self.get_location().clone();
        if source != destination {
            let source_mut = source.get_ship_mut_read();
            let mut source_container = source.get_unit_container_write();
            let mut self_mut = self.mutables.write().unwrap();
            let destination_mut = destination.get_ship_mut_read();
            let mut destination_container = destination.get_unit_container_write();
            if self_mut.hull.get() > 0
                && source_mut
                    .map(|ship_mut| ship_mut.hull.get() > 0)
                    .unwrap_or(true)
                && destination_mut
                    .map(|ship_mut| ship_mut.hull.get() > 0)
                    .unwrap_or(true)
                && source.check_remove(&source_container, self.get_unit())
                && destination.check_insert(&destination_container, self.get_unit())
                && self.acyclicity_check(destination.clone())
            {
                source.remove_unit(&mut source_container, self.get_unit());
                self_mut.location = destination.clone();
                match destination.clone() {
                    UnitLocation::Hangar(hangar) => self_mut.last_mother = Some(hangar.mother.id),
                    UnitLocation::Squadron(squadron) => self_mut.last_mother = Some(squadron.id),
                    _ => {}
                }
                destination.insert_unit(&mut destination_container, self.get_unit());
                true
            } else {
                false
            }
        } else {
            false
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        let location = self.get_mother_node();
        let mutables = self.mutables.read().unwrap();
        if mutables.movement_left > 0 {
            if let Some((viable, _speed)) = mutables
                .engines
                .iter()
                .find_map(|e| e.check_engine(root, location.clone(), destinations))
            {
                Some(viable)
            } else {
                None
            }
        } else {
            None
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| {
                let (active, passive): (Vec<Unit>, Vec<Unit>) = hangar
                    .unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .cloned()
                    .partition(|ship| {
                        ship.destinations_check(root, &vec![destination.clone()])
                            .is_some()
                    });
                let active_daughters = active
                    .iter()
                    .map(|unit| {
                        let mut active_daughters_mut = unit.get_daughters();
                        active_daughters_mut.insert(0, unit.clone());
                        active_daughters_mut
                    })
                    .collect::<Vec<Vec<Unit>>>();
                let passive_daughters = passive
                    .iter()
                    .map(|unit| {
                        let mut passive_daughters_mut =
                            unit.get_traversal_checked_daughters(root, destination.clone());
                        passive_daughters_mut.insert(0, unit.clone());
                        passive_daughters_mut
                    })
                    .collect::<Vec<Vec<Unit>>>();
                vec![active_daughters, passive_daughters]
            })
            .flatten()
            .flatten()
            .flatten()
            .collect()
    }
    fn set_movement_recursive(&self, value: u64) {
        let mut mutables = self.mutables.write().unwrap();
        let new_value = mutables.movement_left.saturating_sub(value);
        mutables.movement_left = new_value;
        drop(mutables);
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.set_movement_recursive(value));
    }
    fn get_moves_left(&self, turn: u64) -> u64 {
        if let Some(engine) = self
            .mutables
            .read()
            .unwrap()
            .engines
            .iter()
            .find(|engine| engine.check_engine_movement_only(turn))
        {
            engine.get_moves_left(self.mutables.read().unwrap().movement_left)
        } else {
            0
        }
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        let location = self.get_mother_node();
        let mut mutables = self.mutables.write().unwrap();
        let movement_left_old = mutables.movement_left;
        match movement_left_old > 0 {
            true => {
                if let Some(speed) = mutables
                    .engines
                    .iter_mut()
                    .find_map(|e| e.process_engine(root, location.clone(), destination.clone()))
                {
                    mutables.movement_left =
                        mutables.movement_left.saturating_sub(u64::MAX / speed);
                } else {
                    panic!();
                }
            }
            false => {
                panic!();
            }
        }
        let movement_left = mutables.movement_left;
        drop(mutables);
        self.set_movement_recursive(movement_left.saturating_sub(movement_left_old));
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        self.process_engines(root, destination.clone());
        self.transfer(UnitLocation::Node(destination.clone()));
        Some(destination)
    }
    fn calculate_damage(
        &self,
        root: &Root,
        is_victor: bool,
        allied_strength: f32,
        enemy_strength: f32,
        duration: u64,
        duration_damage_rand: f32,
        rng: &mut Hc128Rng,
    ) -> (i64, Vec<i64>, Vec<i64>) {
        //to calculate how much damage the ship takes
        //we first have a multiplicative damage value, which is:
        //the ship's maximum health
        //times the ratio between the enemy's strength and the ship's coalition's strength
        //times the battle's duration as a fraction of the 'typical' battle duration, modified by a modder-specified scalar and a battle-wide random value
        //times a ship-specific random factor
        //times the modder-specified multiplier for damage taken by winning or losing ships
        //
        //then we add to that an additive damage value, which is:
        //the modder-defined base damage value
        //times the strength ratio
        //times the duration modifier
        //times the random factor
        //times the losing-ship multiplier
        //
        //then we divide all that by the sum of the ship's toughness and escape scalars
        let rand_factor = Normal::<f32>::new(0.25, root.config.battle_scalars.damage_dev)
            .unwrap()
            .sample(rng)
            .clamp(0.0, 10.0);
        let vae = match is_victor {
            true => root.config.battle_scalars.vae_victor,
            false => root.config.battle_scalars.vae_victis,
        };
        //we do basically the same thing for winning ships and losing ships
        //except that the strength ratio is reversed
        //we use the damage multiplier for winners or losers
        //and we don't take battleescapescalar into account for winners
        let damage = (((self.class.base_hull as f32
            * (enemy_strength / allied_strength)
            * ((duration as f32 / root.config.battle_scalars.avg_duration as f32)
                * root.config.battle_scalars.duration_damage_scalar
                * duration_damage_rand)
            * rand_factor
            * vae)
            + (root.config.battle_scalars.base_damage
                * (enemy_strength / allied_strength)
                * ((duration as f32 / root.config.battle_scalars.avg_duration as f32)
                    * root.config.battle_scalars.duration_damage_scalar
                    * duration_damage_rand)
                * rand_factor
                * vae))
            / (self.class.toughness_scalar
                + (self.class.battle_escape_scalar * (!is_victor as i8) as f32)))
            as i64;
        let engine_damage: Vec<i64> = self
            .mutables
            .read()
            .unwrap()
            .engines
            .iter()
            .filter(|e| e.health.is_some())
            .map(|e| {
                ((damage as f32
                    * Normal::<f32>::new(1.0, root.config.battle_scalars.damage_dev)
                        .unwrap()
                        .sample(rng)
                        .clamp(0.0, 2.0)
                    * root.config.battle_scalars.engine_damage_scalar)
                    / e.class.toughness_scalar) as i64
            })
            .collect();
        let subsystem_damage: Vec<i64> = self
            .mutables
            .read()
            .unwrap()
            .subsystems
            .iter()
            .filter(|s| s.health.is_some())
            .map(|s| {
                ((damage as f32
                    * Normal::<f32>::new(1.0, root.config.battle_scalars.damage_dev)
                        .unwrap()
                        .sample(rng)
                        .clamp(0.0, 2.0)
                    * root.config.battle_scalars.strategic_weapon_damage_scalar)
                    / s.class.toughness_scalar) as i64
            })
            .collect();
        (damage, engine_damage, subsystem_damage)
    }
    fn damage(&self, damage: i64, engine_damage: &Vec<i64>, subsystem_damage: &Vec<i64>) {
        let mut mutables = self.mutables.write().unwrap();
        let current_hull = mutables.hull.get();
        mutables
            .hull
            .set(((current_hull as i64) - (damage)).clamp(0, self.class.base_hull as i64) as u64);
        engine_damage
            .iter()
            .zip(
                mutables
                    .engines
                    .iter_mut()
                    .filter(|engine| engine.health.is_some()),
            )
            .for_each(|(d, e)| {
                e.health = Some(
                    (e.health.unwrap() as i64 - *d).clamp(0, e.class.base_health.unwrap() as i64)
                        as u64,
                );
            });
        subsystem_damage
            .iter()
            .zip(
                mutables
                    .subsystems
                    .iter_mut()
                    .filter(|s| s.health.is_some()),
            )
            .for_each(|(d, s)| {
                s.health = Some(
                    (s.health.unwrap() as i64 - *d).clamp(0, s.class.base_health.unwrap() as i64)
                        as u64,
                );
            });
    }
    fn repair(&self, per_engagement: bool) {
        let mut mutables: RwLockWriteGuard<ShipMut> = self.mutables.write().unwrap();
        let current_hull = mutables.hull.get();
        if (current_hull > 0 && current_hull < self.class.base_hull)
            || mutables
                .engines
                .iter()
                .any(|e| e.health < e.class.base_health)
        {
            let [(hull_repair_points, hull_repair_factor), (engine_repair_points, engine_repair_factor), (subsystem_repair_points, subsystem_repair_factor)]: [(i64, f32); 3] = mutables
                .repairers
                .iter_mut()
                .filter(|rp| rp.class.per_engagement == per_engagement)
                .filter(|rp| rp.get_state() == FactoryState::Active)
                .map(|rp| {
                    rp.process();
                    [
                        (rp.class.repair_points, rp.class.repair_factor),
                        (rp.class.engine_repair_points, rp.class.engine_repair_factor),
                        (rp.class.subsystem_repair_points, rp.class.subsystem_repair_factor),
                    ]
                })
                .fold([(0, 0.0); 3], |a, b| {
                    [
                        (a[0].0 + b[0].0, a[0].1 + b[0].1),
                        (a[1].0 + b[1].0, a[1].1 + b[1].1),
                        (a[2].0 + b[2].0, a[2].1 + b[2].1),
                    ]
                });
            mutables.hull.set(
                (current_hull as i64
                    + hull_repair_points
                    + (self.class.base_hull as f32 * hull_repair_factor) as i64)
                    .clamp(0, self.class.base_hull as i64) as u64,
            );
            mutables
                .engines
                .iter_mut()
                .filter(|e| e.health.is_some())
                .for_each(|e| {
                    (e.health.unwrap() as i64
                        + engine_repair_points
                        + (e.class.base_health.unwrap() as f32 * engine_repair_factor) as i64)
                        .clamp(0, e.class.base_health.unwrap() as i64) as u64;
                });
            mutables
                .subsystems
                .iter_mut()
                .filter(|s| s.health.is_some())
                .for_each(|s| {
                    (s.health.unwrap() as i64
                        + subsystem_repair_points
                        + (s.class.base_health.unwrap() as f32 * subsystem_repair_factor) as i64)
                        .clamp(0, s.class.base_health.unwrap() as i64) as u64;
                });
        }
    }
    fn kill(&self) {
        self.mutables.write().unwrap().hull.set(0);
        self.get_daughters().iter().for_each(|ship| ship.kill());
    }
    fn is_alive(&self) -> bool {
        self.mutables.read().unwrap().hull.get() > 0
    }
    fn is_alive_locked(
        &self,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        _squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        _container_is_not_empty_map: &HashMap<u64, bool>,
    ) -> bool {
        ships_mut_lock
            .get(&self.id)
            .expect(
                format!(
                    "Index {} for ship {} not found in ships_mut_lock.",
                    self.id, self.visible_name
                )
                .as_str(),
            )
            .hull
            .get()
            > 0
    }
    fn record(&self) -> UnitRecord {
        UnitRecord {
            id: self.id,
            visible_name: self.visible_name.clone(),
            class: ShipClass::get_unitclass(self.class.clone()),
            allegiance: self.mutables.read().unwrap().allegiance.clone(),
            daughters: self
                .get_daughters()
                .iter()
                .map(|unit| unit.get_id())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquadronFlavor {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
}

impl PartialEq for SquadronFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SquadronFlavor {}

impl Ord for SquadronFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for SquadronFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for SquadronFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SquadronClassID {
    pub index: usize,
}

impl SquadronClassID {
    pub fn new_from_index(index: usize) -> Self {
        SquadronClassID { index: index }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SquadronClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub squadronflavor: Arc<SquadronFlavor>,
    pub visibility: bool,
    pub capacity: u64,
    pub target: u64,
    pub propagates: bool,
    pub strength_mod: (f32, u64),
    pub allowed: Option<Vec<UnitClassID>>,
    pub ideal: HashMap<UnitClassID, u64>,
    pub sub_target_supply_scalar: f32, //multiplier used for demand generated by non-ideal ships under the target limit; should be below one
    pub non_ideal_demand_scalar: f32, //multiplier used for demand generated for non-ideal unitclasses; should be below one
    pub nav_quorum: f32,
    pub creation_threshold: f32,
    pub de_ghost_threshold: f32,
    pub disband_threshold: f32,
    pub deploys_self: bool, //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments
    pub mother_loyalty_scalar: f32,
    pub defect_chance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub defect_escape_mod: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this squadronclass to be
}

impl SquadronClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.ideal
            .iter()
            .map(|(unitclassid, v)| unitclassid.get_unitclass(root).get_ideal_strength(root) * v)
            .sum()
    }
    pub fn get_ideal_volume(&self) -> u64 {
        self.target
    }
    pub fn get_unitclass(class: Arc<Self>) -> UnitClass {
        UnitClass::SquadronClass(class.clone())
    }
    pub fn instantiate(
        class: Arc<Self>,
        location: UnitLocation,
        faction: Arc<Faction>,
        root: &Root,
    ) -> Squadron {
        let index = root.unit_counter.fetch_add(1, atomic::Ordering::Relaxed);
        Squadron {
            id: index,
            visible_name: uuid::Uuid::new_v4().to_string(),
            class: class.clone(),
            ideal_strength: class.get_ideal_strength(root),
            mutables: RwLock::new(SquadronMut {
                visibility: class.visibility,
                location: location,
                allegiance: faction,
                last_mother: None,
                objectives: Vec::new(),
                ghost: true,
            }),
            unit_container: RwLock::new(UnitContainer::new()),
        }
    }
}

impl Eq for SquadronClass {}

impl Ord for SquadronClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for SquadronClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for SquadronClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

//NOTE: At present, this is an exact copy of ShipAI except that it doesn't have an ID, because it's always calculated on the fly.
#[derive(Debug, Clone)]
pub struct NavAI {
    pub nav_threshold: f32,
    pub ship_attract_specific: f32,
    pub ship_attract_generic: f32,
    pub ship_cargo_attract: HashMap<UnitClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct SquadronMut {
    pub visibility: bool,
    pub location: UnitLocation,
    pub allegiance: Arc<Faction>,
    pub last_mother: Option<u64>,
    pub objectives: Vec<Objective>,
    pub ghost: bool,
}

#[derive(Debug)]
pub struct Squadron {
    pub id: u64,
    pub visible_name: String,
    pub class: Arc<SquadronClass>,
    pub ideal_strength: u64,
    pub mutables: RwLock<SquadronMut>,
    pub unit_container: RwLock<UnitContainer>,
}

impl Squadron {
    pub fn get_unitclass_supply_local(
        &self,
        unit_container: &RwLockWriteGuard<UnitContainer>,
        unitclass: UnitClass,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
    ) -> u64 {
        let unitclass_id = UnitClassID::new_from_unitclass(&unitclass);
        if self
            .class
            .allowed
            .as_ref()
            .map(|allowed_vec| allowed_vec.contains(&unitclass_id))
            .unwrap_or(true)
        {
            unit_container
                .contents
                .iter()
                .filter(|unit| {
                    unit.get_ship()
                        .map(|ship| ships_mut_lock.get(&ship.id).unwrap().hull.get() > 0)
                        .unwrap_or(true)
                })
                .filter(|daughter| &daughter.get_unitclass() == &unitclass)
                .map(|daughter| {
                    daughter.get_real_volume_locked(squadrons_containers_lock, ships_mut_lock)
                })
                .sum()
        } else {
            0
        }
    }
    pub fn get_unitclass_demand_local(
        &self,
        unit_container: &RwLockWriteGuard<UnitContainer>,
        unitclass: UnitClass,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
    ) -> u64 {
        let unitclass_id = UnitClassID::new_from_unitclass(&unitclass);
        if self
            .class
            .allowed
            .as_ref()
            .map(|allowed_vec| allowed_vec.contains(&unitclass_id))
            .unwrap_or(true)
        {
            let daughters = &unit_container.contents;
            let daughter_unitclass_volume = daughters
                .iter()
                .filter(|unit| {
                    unit.get_ship()
                        .map(|ship| ships_mut_lock.get(&ship.id).unwrap().hull.get() > 0)
                        .unwrap_or(true)
                })
                .filter(|unit| &unit.get_unitclass() == &unitclass)
                .map(|unit| unit.get_real_volume_locked(squadrons_containers_lock, ships_mut_lock))
                .sum::<u64>();
            let ideal_volume = self
                .class
                .ideal
                .get(&UnitClassID::new_from_unitclass(&unitclass))
                .unwrap_or(&0)
                * unitclass.get_ideal_volume();
            let ideal_demand = ideal_volume.saturating_sub(daughter_unitclass_volume);
            let non_ideal_demand = (self
                .class
                .target
                .saturating_sub(
                    daughters
                        .iter()
                        .map(|daughter| {
                            daughter
                                .get_real_volume_locked(squadrons_containers_lock, ships_mut_lock)
                        })
                        .sum(),
                )
                .saturating_sub(ideal_demand) as f32
                * self.class.non_ideal_demand_scalar) as u64;
            ideal_demand + non_ideal_demand
        } else {
            0
        }
    }
}

impl PartialEq for Squadron {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.visible_name == other.visible_name
            && self.class == other.class
            && self.ideal_strength == other.ideal_strength
            && self.mutables.read().unwrap().clone() == other.mutables.read().unwrap().clone()
    }
}

impl Eq for Squadron {}

impl Ord for Squadron {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Squadron {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Squadron {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Mobility for Arc<Squadron> {
    fn get_unit(&self) -> Unit {
        Unit::Squadron(self.clone())
    }
    fn get_unitclass(&self) -> UnitClass {
        UnitClass::SquadronClass(self.class.clone())
    }
    fn get_ship(&self) -> Option<Arc<Ship>> {
        None
    }
    fn get_squadron(&self) -> Option<Arc<Squadron>> {
        Some(self.clone())
    }
    fn get_id(&self) -> u64 {
        self.id
    }
    fn get_visible_name(&self) -> String {
        self.visible_name.clone()
    }
    fn is_ship(&self) -> bool {
        false
    }
    fn get_location(&self) -> UnitLocation {
        self.mutables.read().unwrap().location.clone()
    }
    fn check_location_coherency(&self) {
        let mother = self.get_location();
        let sisters: Vec<_> = match mother.clone() {
            UnitLocation::Node(node) => node
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Squadron(squadron) => squadron
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Hangar(hangar) => hangar
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
        };
        assert_eq!(
            sisters
                .iter()
                .filter(|sister| sister.get_id() == self.id)
                .count(),
            1
        );
    }
    fn get_base_hull(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_base_hull())
            .sum()
    }
    fn get_hull(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_hull())
            .sum()
    }
    fn get_engine_base_health(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_engine_base_health())
            .sum()
    }
    fn get_subsystem_base_health(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_subsystem_base_health())
            .sum()
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        self.mutables.read().unwrap().allegiance.clone()
    }
    fn get_daughters(&self) -> Vec<Unit> {
        self.unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .cloned()
            .collect()
    }
    fn get_daughters_recursive(&self) -> Vec<Unit> {
        iter::once(self.clone().get_unit())
            .chain(
                self.unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .map(|daughter| daughter.get_daughters_recursive())
                    .flatten(),
            )
            .collect()
    }
    fn get_undocked_daughters(&self) -> Vec<Arc<Ship>> {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_undocked_daughters())
            .flatten()
            .collect()
    }
    fn get_mother_loyalty_scalar(&self) -> f32 {
        self.class.mother_loyalty_scalar
    }
    fn get_morale_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_morale_scalar())
            .product()
    }
    fn get_character_strength_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_character_strength_scalar())
            .product()
    }
    fn get_interdiction_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_interdiction_scalar())
            .product()
    }
    fn get_processor_demand_nav_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_processor_demand_nav_scalar())
            .product()
    }
    fn get_strength(&self, time: u64) -> u64 {
        let (factor, additive) = self.class.strength_mod;
        let sum = self
            .get_daughters()
            .iter()
            .map(|daughter| daughter.get_strength(time))
            .sum::<u64>();
        (sum as f32 * factor) as u64 + additive
    }
    fn get_strength_locked(
        &self,
        hangars_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
        time: u64,
    ) -> u64 {
        let (factor, additive) = self.class.strength_mod;
        let sum = squadrons_containers_lock
            .get(&self.id)
            .unwrap()
            .contents
            .iter()
            .filter(|daughter| {
                daughter.is_alive_locked(
                    ships_mut_lock,
                    squadrons_mut_lock,
                    container_is_not_empty_map,
                )
            })
            .map(|daughter| {
                daughter.get_strength_locked(
                    hangars_containers_lock,
                    squadrons_containers_lock,
                    ships_mut_lock,
                    squadrons_mut_lock,
                    container_is_not_empty_map,
                    time,
                )
            })
            .sum::<u64>();
        (sum as f32 * factor) as u64 + additive
    }
    fn get_strength_post_engagement(&self, _damage: i64) -> u64 {
        0
    }
    fn get_real_volume(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_real_volume())
            .sum()
    }
    fn get_real_volume_locked(
        &self,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
    ) -> u64 {
        squadrons_containers_lock
            .get(&self.id)
            .expect(
                format!(
                    "squadrons_containers_lock does not contain index {} for squadron {}",
                    self.id, self.visible_name
                )
                .as_str(),
            )
            .contents
            .iter()
            .filter(|unit| {
                unit.get_ship()
                    .map(|ship| ships_mut_lock.get(&ship.id).unwrap().hull.get() > 0)
                    .unwrap_or(true)
            })
            .map(|daughter| {
                daughter.get_real_volume_locked(squadrons_containers_lock, ships_mut_lock)
            })
            .sum()
    }
    fn get_capacity_volume(&self) -> u64 {
        self.class.capacity
    }
    fn get_ai(&self) -> NavAI {
        let daughters = self.get_daughters();
        let daughters_len = daughters.len() as f32;
        daughters.iter().fold(
            NavAI {
                nav_threshold: 0.0,
                ship_attract_specific: 0.0,
                ship_attract_generic: 0.0,
                ship_cargo_attract: HashMap::new(),
                resource_attract: HashMap::new(),
                friendly_supply_attract: 0.0,
                hostile_supply_attract: 0.0,
                allegiance_demand_attract: 0.0,
                enemy_demand_attract: 0.0,
                strategic_weapon_damage_attract: 0.0,
                strategic_weapon_engine_damage_attract: 0.0,
                strategic_weapon_subsystem_damage_attract: 0.0,
                strategic_weapon_healing_attract: 0.0,
                strategic_weapon_engine_healing_attract: 0.0,
                strategic_weapon_subsystem_healing_attract: 0.0,
            },
            |mut acc, ship| {
                let sub_ai = ship.get_ai();
                acc.nav_threshold += sub_ai.nav_threshold / daughters_len;
                acc.ship_attract_specific += sub_ai.ship_attract_specific / daughters_len;
                acc.ship_attract_generic += sub_ai.ship_attract_generic / daughters_len;
                sub_ai.ship_cargo_attract.iter().for_each(|(scid, scalar)| {
                    *acc.ship_cargo_attract.entry(*scid).or_insert(0.0) += scalar / daughters_len;
                });
                sub_ai
                    .resource_attract
                    .iter()
                    .for_each(|(resource, scalar)| {
                        *acc.resource_attract.entry(resource.clone()).or_insert(0.0) +=
                            scalar / daughters_len;
                    });
                acc.friendly_supply_attract += sub_ai.friendly_supply_attract / daughters_len;
                acc.hostile_supply_attract += sub_ai.hostile_supply_attract / daughters_len;
                acc.allegiance_demand_attract += sub_ai.allegiance_demand_attract / daughters_len;
                acc.enemy_demand_attract += sub_ai.enemy_demand_attract / daughters_len;
                acc.strategic_weapon_damage_attract +=
                    sub_ai.strategic_weapon_damage_attract / daughters_len;
                acc.strategic_weapon_engine_damage_attract +=
                    sub_ai.strategic_weapon_engine_damage_attract / daughters_len;
                acc.strategic_weapon_subsystem_damage_attract +=
                    sub_ai.strategic_weapon_subsystem_damage_attract / daughters_len;
                acc.strategic_weapon_healing_attract +=
                    sub_ai.strategic_weapon_healing_attract / daughters_len;
                acc.strategic_weapon_engine_healing_attract +=
                    sub_ai.strategic_weapon_engine_healing_attract / daughters_len;
                acc.strategic_weapon_subsystem_healing_attract +=
                    sub_ai.strategic_weapon_subsystem_healing_attract / daughters_len;
                acc
            },
        )
    }
    fn get_nav_threshold(&self) -> f32 {
        self.get_ai().nav_threshold.clone()
    }
    fn get_objectives(&self) -> Vec<Objective> {
        self.mutables
            .read()
            .unwrap()
            .objectives
            .iter()
            .cloned()
            .collect()
    }
    fn get_deployment_threshold(&self) -> Option<u64> {
        self.class.deploys_daughters
    }
    fn get_deployment_status(&self) -> bool {
        self.class.deploys_self
    }
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32) {
        (
            self.class.defect_chance.clone(),
            self.get_daughters()
                .iter()
                .map(|unit| unit.get_defection_data().1)
                .product::<f32>()
                * self.class.defect_escape_mod,
        )
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_supply(resource.clone()))
            .sum()
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand_from_stockpiles(resource.clone()))
            .sum()
    }
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand_from_processors(resource.clone()))
            .sum()
    }
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        (&self.get_unitclass() == &unitclass) as u64
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_unitclass_num(unitclass.clone()))
                .sum::<u64>()
    }
    fn get_unitclass_supply_recursive(&self, unitclass: UnitClass) -> u64 {
        let daughter_supply = self
            .get_daughters()
            .iter()
            .map(|daughter| daughter.get_unitclass_supply_recursive(unitclass.clone()))
            .sum::<u64>();
        let ideal_volume = self
            .class
            .ideal
            .get(&UnitClassID::new_from_unitclass(&unitclass))
            .unwrap_or(&0)
            * unitclass.get_ideal_volume();
        let non_ideal_volume = daughter_supply.saturating_sub(ideal_volume);
        let excess_volume = self.get_real_volume().saturating_sub(self.class.target);
        let over_target_supply = (excess_volume).min(non_ideal_volume);
        let under_target_supply = ((non_ideal_volume.saturating_sub(over_target_supply)) as f32
            * self.class.sub_target_supply_scalar) as u64;
        over_target_supply + under_target_supply
    }
    fn get_unitclass_demand_recursive(&self, unitclass: UnitClass) -> u64 {
        let daughter_volume = self
            .get_daughters()
            .iter()
            .filter(|unit| &unit.get_unitclass() == &unitclass)
            .map(|unit| unit.get_real_volume())
            .sum::<u64>();
        let ideal_volume = self
            .class
            .ideal
            .get(&UnitClassID::new_from_unitclass(&unitclass))
            .unwrap_or(&0)
            * unitclass.get_ideal_volume();
        ideal_volume.saturating_sub(daughter_volume)
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_unitclass_demand_recursive(unitclass.clone()))
                .sum::<u64>()
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        self.mutables.write().unwrap().allegiance = new_faction.clone();
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.change_allegiance(new_faction.clone()));
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        let source = self.get_location().clone();
        if source != destination {
            let source_mut = source.get_ship_mut_read();
            let mut source_container = source.get_unit_container_write();
            let mut self_mut = self.mutables.write().unwrap();
            let destination_mut = destination.get_ship_mut_read();
            let mut destination_container = destination.get_unit_container_write();
            if source_mut
                .map(|ship_mut| ship_mut.hull.get() > 0)
                .unwrap_or(true)
                && destination_mut
                    .map(|ship_mut| ship_mut.hull.get() > 0)
                    .unwrap_or(true)
                && source.check_remove(&source_container, self.get_unit())
                && destination.check_insert(&destination_container, self.get_unit())
                && self.acyclicity_check(destination.clone())
            {
                source.remove_unit(&mut source_container, self.get_unit());
                self_mut.location = destination.clone();
                match destination.clone() {
                    UnitLocation::Hangar(hangar) => self_mut.last_mother = Some(hangar.mother.id),
                    UnitLocation::Squadron(squadron) => self_mut.last_mother = Some(squadron.id),
                    _ => {}
                }
                destination.insert_unit(&mut destination_container, self.get_unit());
                true
            } else {
                false
            }
        } else {
            false
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        let viable: Vec<_> = destinations
            .iter()
            .filter(|node| {
                self.get_traversal_checked_daughters(root, node.clone().clone())
                    .len()
                    > 0
            })
            .cloned()
            .collect();
        if !viable.is_empty() {
            Some(viable)
        } else {
            None
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        let daughters = self.get_daughters();
        let (passed_ships, failed_ships): (Vec<Unit>, Vec<Unit>) =
            daughters.iter().cloned().partition(|daughter| {
                daughter
                    .destinations_check(root, &vec![destination.clone()])
                    .is_some()
            });
        //we see what fraction of the squadron's strength is unable to make the jump
        //by checking strength of failed ships, and then all daughters
        //we don't just call get_strength on the squadron itself
        //if we did, the squadron's strength modifiers would be counted only toward its total
        let failed_strength = failed_ships
            .iter()
            .map(|ship| ship.get_strength(root.config.battle_scalars.avg_duration) as f32)
            .sum::<f32>();
        let total_strength = daughters
            .iter()
            .map(|daughter| daughter.get_strength(root.config.battle_scalars.avg_duration) as f32)
            .sum::<f32>();
        if (failed_strength / total_strength) < (1.0 - self.class.nav_quorum) {
            passed_ships
        } else {
            Vec::new()
        }
    }
    fn set_movement_recursive(&self, value: u64) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.set_movement_recursive(value));
    }
    fn get_moves_left(&self, turn: u64) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_moves_left(turn))
            .min()
            .unwrap_or(0)
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.process_engines(root, destination.clone()));
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        let daughters = self.get_daughters();
        let valid = self.get_traversal_checked_daughters(root, destination.clone());
        if valid.len() > 0 {
            let left_behind = daughters
                .iter()
                .filter(|daughter| !valid.contains(daughter))
                .collect::<Vec<_>>();
            left_behind.iter().for_each(|expelled| {
                expelled.transfer(UnitLocation::Node(self.get_mother_node()));
            });
            self.process_engines(root, destination.clone());
            self.transfer(UnitLocation::Node(destination.clone()));
            Some(destination)
        } else {
            None
        }
    }
    fn calculate_damage(
        &self,
        _root: &Root,
        _is_victor: bool,
        _victor_strength: f32,
        _victis_strength: f32,
        _duration: u64,
        _duration_damage_rand: f32,
        _rng: &mut Hc128Rng,
    ) -> (i64, Vec<i64>, Vec<i64>) {
        (0, Vec::new(), Vec::new())
    }
    fn damage(&self, _damage: i64, _engine_damage: &Vec<i64>, _subsystem_damage: &Vec<i64>) {}
    fn repair(&self, _per_engagement: bool) {}
    fn kill(&self) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.kill());
    }
    fn is_alive(&self) -> bool {
        self.mutables.read().unwrap().ghost || (!self.get_daughters().is_empty())
    }
    fn is_alive_locked(
        &self,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
    ) -> bool {
        squadrons_mut_lock.get(&self.id).unwrap().ghost
            || *container_is_not_empty_map.get(&self.id).unwrap()
    }
    fn record(&self) -> UnitRecord {
        UnitRecord {
            id: self.id,
            visible_name: self.visible_name.clone(),
            class: SquadronClass::get_unitclass(self.class.clone()),
            allegiance: self.mutables.read().unwrap().allegiance.clone(),
            daughters: self
                .get_daughters()
                .iter()
                .map(|unit| unit.get_id())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum UnitClassID {
    ShipClass(ShipClassID),
    SquadronClass(SquadronClassID),
}

impl UnitClassID {
    pub fn new_from_unitclass(unitclass: &UnitClass) -> Self {
        match unitclass {
            UnitClass::ShipClass(sc) => UnitClassID::ShipClass(ShipClassID::new_from_index(sc.id)),
            UnitClass::SquadronClass(fc) => {
                UnitClassID::SquadronClass(SquadronClassID::new_from_index(fc.id))
            }
        }
    }
    pub fn get_index(&self) -> usize {
        match self {
            UnitClassID::ShipClass(sc) => sc.index,
            UnitClassID::SquadronClass(fc) => fc.index,
        }
    }
    pub fn get_unitclass(&self, root: &Root) -> UnitClass {
        match self {
            UnitClassID::ShipClass(sc) => UnitClass::ShipClass(
                root.shipclasses
                    .iter()
                    .find(|shipclass| shipclass.id == sc.index)
                    .unwrap()
                    .clone(),
            ),
            UnitClassID::SquadronClass(fc) => UnitClass::SquadronClass(
                root.squadronclasses
                    .iter()
                    .find(|squadronclass| squadronclass.id == fc.index)
                    .unwrap()
                    .clone(),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum UnitClass {
    ShipClass(Arc<ShipClass>),
    SquadronClass(Arc<SquadronClass>),
}

impl UnitClass {
    pub fn get_id(&self) -> usize {
        match self {
            UnitClass::ShipClass(sc) => sc.id,
            UnitClass::SquadronClass(fc) => fc.id,
        }
    }
    pub fn get_visible_name(&self) -> String {
        match self {
            UnitClass::ShipClass(sc) => sc.visible_name.clone(),
            UnitClass::SquadronClass(fc) => fc.visible_name.clone(),
        }
    }
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        match self {
            UnitClass::ShipClass(sc) => sc.get_ideal_strength(root),
            UnitClass::SquadronClass(fc) => fc.get_ideal_strength(root),
        }
    }
    pub fn get_ideal_volume(&self) -> u64 {
        match self {
            UnitClass::ShipClass(sc) => sc.get_ideal_volume(),
            UnitClass::SquadronClass(fc) => fc.get_ideal_volume(),
        }
    }
    pub fn get_value_mult(&self) -> f32 {
        match self {
            UnitClass::ShipClass(sc) => sc.value_mult,
            UnitClass::SquadronClass(fc) => fc.value_mult,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Unit {
    Ship(Arc<Ship>),
    Squadron(Arc<Squadron>),
}

impl Mobility for Unit {
    fn get_unit(&self) -> Unit {
        self.clone()
    }
    fn get_unitclass(&self) -> UnitClass {
        match self {
            Unit::Ship(ship) => ship.get_unitclass(),
            Unit::Squadron(squadron) => squadron.get_unitclass(),
        }
    }
    fn get_ship(&self) -> Option<Arc<Ship>> {
        match self {
            Unit::Ship(ship) => Some(ship.clone()),
            _ => None,
        }
    }
    fn get_squadron(&self) -> Option<Arc<Squadron>> {
        match self {
            Unit::Squadron(squadron) => Some(squadron.clone()),
            _ => None,
        }
    }
    fn get_id(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_id(),
            Unit::Squadron(squadron) => squadron.get_id(),
        }
    }
    fn get_visible_name(&self) -> String {
        match self {
            Unit::Ship(ship) => ship.get_visible_name(),
            Unit::Squadron(squadron) => squadron.get_visible_name(),
        }
    }
    fn is_ship(&self) -> bool {
        match self {
            Unit::Ship(ship) => ship.is_ship(),
            Unit::Squadron(squadron) => squadron.is_ship(),
        }
    }
    fn get_location(&self) -> UnitLocation {
        match self {
            Unit::Ship(ship) => ship.get_location(),
            Unit::Squadron(squadron) => squadron.get_location(),
        }
    }
    fn check_location_coherency(&self) {
        match self {
            Unit::Ship(ship) => ship.check_location_coherency(),
            Unit::Squadron(squadron) => squadron.check_location_coherency(),
        }
    }
    fn get_base_hull(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_base_hull(),
            Unit::Squadron(squadron) => squadron.get_base_hull(),
        }
    }
    fn get_hull(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_hull(),
            Unit::Squadron(squadron) => squadron.get_hull(),
        }
    }
    fn get_engine_base_health(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_engine_base_health(),
            Unit::Squadron(squadron) => squadron.get_engine_base_health(),
        }
    }
    fn get_subsystem_base_health(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_subsystem_base_health(),
            Unit::Squadron(squadron) => squadron.get_subsystem_base_health(),
        }
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        match self {
            Unit::Ship(ship) => ship.get_allegiance(),
            Unit::Squadron(squadron) => squadron.get_allegiance(),
        }
    }
    fn get_daughters(&self) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_daughters(),
            Unit::Squadron(squadron) => squadron.get_daughters(),
        }
    }
    fn get_daughters_recursive(&self) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_daughters_recursive(),
            Unit::Squadron(squadron) => squadron.get_daughters_recursive(),
        }
    }
    fn get_mother_loyalty_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_mother_loyalty_scalar(),
            Unit::Squadron(squadron) => squadron.get_mother_loyalty_scalar(),
        }
    }
    fn get_undocked_daughters(&self) -> Vec<Arc<Ship>> {
        match self {
            Unit::Ship(ship) => ship.get_undocked_daughters(),
            Unit::Squadron(squadron) => squadron.get_undocked_daughters(),
        }
    }
    fn get_morale_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_morale_scalar(),
            Unit::Squadron(squadron) => squadron.get_morale_scalar(),
        }
    }
    fn get_character_strength_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_character_strength_scalar(),
            Unit::Squadron(squadron) => squadron.get_character_strength_scalar(),
        }
    }
    fn get_interdiction_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_interdiction_scalar(),
            Unit::Squadron(squadron) => squadron.get_interdiction_scalar(),
        }
    }
    fn get_processor_demand_nav_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_processor_demand_nav_scalar(),
            Unit::Squadron(squadron) => squadron.get_processor_demand_nav_scalar(),
        }
    }
    fn get_strength(&self, time: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength(time),
            Unit::Squadron(squadron) => squadron.get_strength(time),
        }
    }
    fn get_strength_locked(
        &self,
        hangars_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
        time: u64,
    ) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength_locked(
                hangars_containers_lock,
                squadrons_containers_lock,
                ships_mut_lock,
                squadrons_mut_lock,
                container_is_not_empty_map,
                time,
            ),
            Unit::Squadron(squadron) => squadron.get_strength_locked(
                hangars_containers_lock,
                squadrons_containers_lock,
                ships_mut_lock,
                squadrons_mut_lock,
                container_is_not_empty_map,
                time,
            ),
        }
    }
    fn get_strength_post_engagement(&self, damage: i64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength_post_engagement(damage),
            Unit::Squadron(squadron) => squadron.get_strength_post_engagement(damage),
        }
    }
    fn get_real_volume(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_real_volume(),
            Unit::Squadron(squadron) => squadron.get_real_volume(),
        }
    }
    fn get_real_volume_locked(
        &self,
        squadrons_containers_lock: &HashMap<u64, RwLockWriteGuard<UnitContainer>>,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
    ) -> u64 {
        match self {
            Unit::Ship(ship) => {
                ship.get_real_volume_locked(squadrons_containers_lock, ships_mut_lock)
            }
            Unit::Squadron(squadron) => {
                squadron.get_real_volume_locked(squadrons_containers_lock, ships_mut_lock)
            }
        }
    }
    fn get_capacity_volume(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_capacity_volume(),
            Unit::Squadron(squadron) => squadron.get_capacity_volume(),
        }
    }
    fn get_ai(&self) -> NavAI {
        match self {
            Unit::Ship(ship) => ship.get_ai(),
            Unit::Squadron(squadron) => squadron.get_ai(),
        }
    }
    fn get_nav_threshold(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_nav_threshold(),
            Unit::Squadron(squadron) => squadron.get_nav_threshold(),
        }
    }
    fn get_objectives(&self) -> Vec<Objective> {
        match self {
            Unit::Ship(ship) => ship.get_objectives(),
            Unit::Squadron(squadron) => squadron.get_objectives(),
        }
    }
    fn get_deployment_threshold(&self) -> Option<u64> {
        match self {
            Unit::Ship(ship) => ship.get_deployment_threshold(),
            Unit::Squadron(squadron) => squadron.get_deployment_threshold(),
        }
    }
    fn get_deployment_status(&self) -> bool {
        match self {
            Unit::Ship(ship) => ship.get_deployment_status(),
            Unit::Squadron(squadron) => squadron.get_deployment_status(),
        }
    }
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32) {
        match self {
            Unit::Ship(ship) => ship.get_defection_data(),
            Unit::Squadron(squadron) => squadron.get_defection_data(),
        }
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_supply(resource),
            Unit::Squadron(squadron) => squadron.get_resource_supply(resource),
        }
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand(resource),
            Unit::Squadron(squadron) => squadron.get_resource_demand(resource),
        }
    }
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand_from_stockpiles(resource),
            Unit::Squadron(squadron) => squadron.get_resource_demand_from_stockpiles(resource),
        }
    }
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand_from_processors(resource),
            Unit::Squadron(squadron) => squadron.get_resource_demand_from_processors(resource),
        }
    }
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_unitclass_num(unitclass),
            Unit::Squadron(squadron) => squadron.get_unitclass_num(unitclass),
        }
    }
    fn get_unitclass_supply_recursive(&self, unitclass: UnitClass) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_unitclass_supply_recursive(unitclass),
            Unit::Squadron(squadron) => squadron.get_unitclass_supply_recursive(unitclass),
        }
    }
    fn get_unitclass_demand_recursive(&self, unitclass: UnitClass) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_unitclass_demand_recursive(unitclass),
            Unit::Squadron(squadron) => squadron.get_unitclass_demand_recursive(unitclass),
        }
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        match self {
            Unit::Ship(ship) => ship.change_allegiance(new_faction),
            Unit::Squadron(squadron) => squadron.change_allegiance(new_faction),
        }
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        match self {
            Unit::Ship(ship) => ship.transfer(destination),
            Unit::Squadron(squadron) => squadron.transfer(destination),
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        match self {
            Unit::Ship(ship) => ship.destinations_check(root, destinations),
            Unit::Squadron(squadron) => squadron.destinations_check(root, destinations),
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_traversal_checked_daughters(root, destination),
            Unit::Squadron(squadron) => squadron.get_traversal_checked_daughters(root, destination),
        }
    }
    fn set_movement_recursive(&self, value: u64) {
        match self {
            Unit::Ship(ship) => ship.set_movement_recursive(value),
            Unit::Squadron(squadron) => squadron.set_movement_recursive(value),
        }
    }
    fn get_moves_left(&self, turn: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_moves_left(turn),
            Unit::Squadron(squadron) => squadron.get_moves_left(turn),
        }
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        match self {
            Unit::Ship(ship) => ship.process_engines(root, destination),
            Unit::Squadron(squadron) => squadron.process_engines(root, destination),
        }
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        match self {
            Unit::Ship(ship) => ship.traverse(root, destination),
            Unit::Squadron(squadron) => squadron.traverse(root, destination),
        }
    }
    fn calculate_damage(
        &self,
        root: &Root,
        is_victor: bool,
        victor_strength: f32,
        victis_strength: f32,
        duration: u64,
        duration_damage_rand: f32,
        rng: &mut Hc128Rng,
    ) -> (i64, Vec<i64>, Vec<i64>) {
        match self {
            Unit::Ship(ship) => ship.calculate_damage(
                root,
                is_victor,
                victor_strength,
                victis_strength,
                duration,
                duration_damage_rand,
                rng,
            ),
            Unit::Squadron(squadron) => squadron.calculate_damage(
                root,
                is_victor,
                victor_strength,
                victis_strength,
                duration,
                duration_damage_rand,
                rng,
            ),
        }
    }
    fn damage(&self, damage: i64, engine_damage: &Vec<i64>, strategic_weapon_damage: &Vec<i64>) {
        match self {
            Unit::Ship(ship) => ship.damage(damage, engine_damage, strategic_weapon_damage),
            Unit::Squadron(squadron) => {
                squadron.damage(damage, engine_damage, strategic_weapon_damage)
            }
        }
    }
    fn repair(&self, per_engagement: bool) {
        match self {
            Unit::Ship(ship) => ship.repair(per_engagement),
            Unit::Squadron(squadron) => squadron.repair(per_engagement),
        }
    }
    fn kill(&self) {
        match self {
            Unit::Ship(ship) => ship.kill(),
            Unit::Squadron(squadron) => squadron.kill(),
        }
    }
    fn is_alive(&self) -> bool {
        match self {
            Unit::Ship(ship) => ship.is_alive(),
            Unit::Squadron(squadron) => squadron.is_alive(),
        }
    }
    fn is_alive_locked(
        &self,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        squadrons_mut_lock: &HashMap<u64, RwLockWriteGuard<SquadronMut>>,
        container_is_not_empty_map: &HashMap<u64, bool>,
    ) -> bool {
        match self {
            Unit::Ship(ship) => ship.is_alive_locked(
                ships_mut_lock,
                squadrons_mut_lock,
                container_is_not_empty_map,
            ),
            Unit::Squadron(squadron) => squadron.is_alive_locked(
                ships_mut_lock,
                squadrons_mut_lock,
                container_is_not_empty_map,
            ),
        }
    }
    fn record(&self) -> UnitRecord {
        match self {
            Unit::Ship(ship) => ship.record(),
            Unit::Squadron(squadron) => squadron.record(),
        }
    }
}
