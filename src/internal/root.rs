use crate::internal::engagement::{EngagementRecord, UnitRecord};
use crate::internal::faction::Faction;
use crate::internal::hangar::HangarClass;
use crate::internal::node::{Cluster, EdgeFlavor, Locality, Node, NodeFlavor};
use crate::internal::resource::{
    EngineClass, FactoryClass, RepairerClass, Resource, ShipyardClass, StrategicWeaponClass,
};
use crate::internal::salience::{polarity::Demand, polarity::Supply, GlobalSalience, Salience};
use crate::internal::unit::{
    Mobility, Ship, ShipAI, ShipClass, ShipFlavor, Squadron, SquadronClass, SquadronFlavor,
    SubsystemClass, Unit, UnitClass, UnitClassID, UnitLocation,
};
use itertools::Itertools;
use rand::prelude::*;
use rand_hc::Hc128Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub salience_scalars: SalienceScalars,
    pub entity_scalars: EntityScalars,
    pub battle_scalars: BattleScalars,
    pub soft_ship_limit: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SalienceScalars {
    pub faction_deg_mult: f32,
    pub resource_deg_mult: f32,
    pub unitclass_deg_mult: f32,
    pub faction_prop_iters: usize, //number of edges across which this salience will propagate during a turn
    pub resource_prop_iters: usize,
    pub unitclass_prop_iters: usize,
    pub volume_strength_ratio: f32, //multiplier for resource/unitclass supply points when comparing to threat values for faction demand calcs
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityScalars {
    pub avg_speed: u64, //typical speed of a cargo ship; that is, the speed divided by cooldown of its main engine
    pub defect_escape_scalar: f32,
    pub victor_morale_scalar: f32,
    pub victis_morale_scalar: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BattleScalars {
    pub avg_duration: u64,
    pub duration_log_exp: f32, //logarithmic exponent for scaling of battle duration over battle size
    pub duration_dev: f32, //standard deviation for the randomly-generated scaling factor for battle duration
    pub attacker_chance_dev: f32, //standard deviation for the randomly-generated scaling factor for the attackers' chance of winning a battle
    pub defender_chance_dev: f32, //standard deviation for the randomly-generated scaling factor for the defenders' chance of winning a battle
    pub vae_victor: f32,          //multiplier for damage done to ships winning a battle
    pub vae_victis: f32,          //multiplier for damage done to ships losing a battle
    pub damage_dev: f32, //standard deviation for the randomly-generated scaling factor for damage done to ships
    pub base_damage: f32, //base value for the additive damage done to ships in addition to the percentage-based damage
    pub engine_damage_scalar: f32,
    pub strategic_weapon_damage_scalar: f32,
    pub duration_damage_scalar: f32, //multiplier for damage increase as battle duration rises
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveTarget {
    Node(Arc<Node>),
    Cluster(Arc<Cluster>),
    Unit(UnitRecord),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveTask {
    Reach(ObjectiveTarget),
    Kill(ObjectiveTarget),
    Protect(ObjectiveTarget),
    Capture(ObjectiveTarget),
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveState {
    Obstructed,
    Complete,
    Incomplete(f32),
    Failed,
    Expired,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Objective {
    pub visible_name: Option<String>,
    pub start_turn: u64,
    pub task: ObjectiveTask,
    pub fraction: Option<f32>,
    pub duration: Option<u64>,
    pub time_limit: Option<u64>,
    pub difficulty: f32,
    pub cost: u64,
    pub duration_scalar: f32,
    pub strength_scalar: f32,
    pub toughness_scalar: f32,
    pub battle_escape_scalar: f32,
    pub required_subgoals: Vec<Objective>,
    pub optional_subgoals: Vec<Objective>,
    pub state: ObjectiveState,
}

impl Objective {
    pub fn update(
        &mut self,
        self_unit: Unit,
        root: &Root,
        propagated_state: Option<ObjectiveState>,
    ) {
        let new_state = match &self.state {
            ObjectiveState::Complete => self.state,
            ObjectiveState::Failed => self.state,
            ObjectiveState::Expired => self.state,
            _ => {
                if let Some(higher_state) = propagated_state {
                    higher_state
                } else {
                    let self_faction = self_unit.get_allegiance();
                    let base_state: ObjectiveState = match &self.task {
                        ObjectiveTask::Reach(target) => match target {
                            ObjectiveTarget::Node(node) => {
                                if self_unit.get_mother_node() == node.clone() {
                                    ObjectiveState::Complete
                                } else {
                                    ObjectiveState::Incomplete(0.0)
                                }
                            }
                            ObjectiveTarget::Cluster(cluster) => {
                                if cluster.nodes.contains(&self_unit.get_mother_node()) {
                                    ObjectiveState::Complete
                                } else {
                                    ObjectiveState::Incomplete(0.0)
                                }
                            }
                            ObjectiveTarget::Unit(target_unit) => {
                                if let Some(target_unit_actual) = target_unit.get_unit(root) {
                                    if self_unit.get_mother_node()
                                        == target_unit_actual.get_mother_node()
                                    {
                                        ObjectiveState::Complete
                                    } else {
                                        ObjectiveState::Incomplete(0.0)
                                    }
                                } else {
                                    ObjectiveState::Failed
                                }
                            }
                        },
                        ObjectiveTask::Kill(target) => match target {
                            ObjectiveTarget::Unit(target_unit) => {
                                if let Some(target_unit_actual) = target_unit.get_unit(root) {
                                    ObjectiveState::Incomplete(
                                        target_unit_actual.get_hull() as f32
                                            / target_unit_actual.get_base_hull() as f32,
                                    )
                                } else {
                                    ObjectiveState::Complete
                                }
                            }
                            _ => panic!(),
                        },
                        ObjectiveTask::Protect(target) => match target {
                            ObjectiveTarget::Node(node) => {
                                let any_enemy_allegiance = self_faction
                                    .get_enemies(root)
                                    .iter()
                                    .any(|enemy| &node.get_allegiance() == enemy);
                                if any_enemy_allegiance {
                                    ObjectiveState::Failed
                                } else {
                                    let completion = self
                                        .duration
                                        .map(|dur| {
                                            dur as f32
                                                / (root.get_turn().saturating_sub(self.start_turn))
                                                    as f32
                                        })
                                        .unwrap_or(1.0);
                                    if completion >= 1.0 {
                                        ObjectiveState::Complete
                                    } else {
                                        ObjectiveState::Incomplete(completion)
                                    }
                                }
                            }
                            ObjectiveTarget::Cluster(cluster) => {
                                let enemy_allegiance_fraction = cluster
                                    .nodes
                                    .iter()
                                    .filter(|node| {
                                        self_faction
                                            .get_enemies(root)
                                            .iter()
                                            .any(|enemy| &node.get_allegiance() == enemy)
                                    })
                                    .count()
                                    as f32
                                    / cluster.nodes.len() as f32;
                                if (1.0 - enemy_allegiance_fraction) > self.fraction.unwrap_or(1.0)
                                {
                                    ObjectiveState::Failed
                                } else {
                                    let completion = self
                                        .duration
                                        .map(|dur| {
                                            dur as f32
                                                / (root.get_turn().saturating_sub(self.start_turn))
                                                    as f32
                                        })
                                        .unwrap_or(1.0);
                                    if completion >= 1.0 {
                                        ObjectiveState::Complete
                                    } else {
                                        ObjectiveState::Incomplete(completion)
                                    }
                                }
                            }
                            ObjectiveTarget::Unit(target_unit) => {
                                if let Some(target_unit_actual) = target_unit.get_unit(root) {
                                    let completion = self
                                        .duration
                                        .map(|dur| {
                                            dur as f32
                                                / (root.get_turn().saturating_sub(self.start_turn))
                                                    as f32
                                        })
                                        .unwrap_or(1.0);
                                    if completion >= 1.0 {
                                        ObjectiveState::Complete
                                    } else {
                                        ObjectiveState::Incomplete(completion)
                                    }
                                } else {
                                    ObjectiveState::Failed
                                }
                            }
                        },
                        //at some point this will need some further complexity to deal with things being held by allied factions, probably
                        ObjectiveTask::Capture(target) => match target {
                            ObjectiveTarget::Node(node) => {
                                if node.get_allegiance() == self_faction {
                                    ObjectiveState::Complete
                                } else {
                                    ObjectiveState::Incomplete(0.0)
                                }
                            }
                            ObjectiveTarget::Cluster(cluster) => {
                                let fraction = cluster
                                    .nodes
                                    .iter()
                                    .filter(|node| node.get_allegiance() == self_faction)
                                    .count() as f32
                                    / cluster.nodes.len() as f32;
                                if fraction >= 1.0 {
                                    ObjectiveState::Complete
                                } else {
                                    ObjectiveState::Incomplete(fraction)
                                }
                            }
                            ObjectiveTarget::Unit(target_unit) => {
                                if let Some(target_unit_actual) = target_unit.get_unit(root) {
                                    if target_unit_actual.get_allegiance() == self_faction {
                                        ObjectiveState::Complete
                                    } else {
                                        ObjectiveState::Incomplete(0.0)
                                    }
                                } else {
                                    ObjectiveState::Failed
                                }
                            }
                        },
                    };
                    let true_state = match base_state {
                        ObjectiveState::Incomplete(_) => {
                            if self
                                .time_limit
                                .map(|dur| root.get_turn() >= self.start_turn + dur)
                                .unwrap_or(false)
                            {
                                ObjectiveState::Expired
                            } else {
                                base_state
                            }
                        }
                        _ => base_state,
                    };
                    true_state
                }
            }
        };

        let state_for_propagation = match new_state {
            ObjectiveState::Complete => Some(new_state),
            ObjectiveState::Failed => Some(new_state),
            ObjectiveState::Expired => Some(new_state),
            _ => None,
        };

        self.required_subgoals
            .iter_mut()
            .for_each(|objective| objective.update(self_unit.clone(), root, state_for_propagation));
        self.optional_subgoals
            .iter_mut()
            .for_each(|objective| objective.update(self_unit.clone(), root, state_for_propagation));

        if !self.required_subgoals.iter().all(|objective| {
            objective.state == ObjectiveState::Complete
                || objective.state == ObjectiveState::Expired
                || objective.state == ObjectiveState::Failed
        }) {
            self.state = ObjectiveState::Obstructed;
        } else {
            self.state = new_state;
        }
    }
}

#[derive(Debug)]
pub struct Root {
    pub config: Config,
    pub nodeflavors: Vec<Arc<NodeFlavor>>,
    pub nodes: Vec<Arc<Node>>,
    pub clusters: Vec<Arc<Cluster>>,
    pub edgeflavors: Vec<Arc<EdgeFlavor>>,
    pub edges: HashMap<(Arc<Node>, Arc<Node>), Arc<EdgeFlavor>>,
    pub neighbors: HashMap<Arc<Node>, Vec<Arc<Node>>>,
    pub factions: Vec<Arc<Faction>>,
    pub wars: HashSet<(Arc<Faction>, Arc<Faction>)>,
    pub resources: Vec<Arc<Resource>>,
    pub hangarclasses: Vec<Arc<HangarClass>>,
    pub hangar_counter: Arc<AtomicU64>,
    pub engineclasses: Vec<Arc<EngineClass>>,
    pub repairerclasses: Vec<Arc<RepairerClass>>,
    pub strategicweaponclasses: Vec<Arc<StrategicWeaponClass>>,
    pub factoryclasses: Vec<Arc<FactoryClass>>,
    pub shipyardclasses: Vec<Arc<ShipyardClass>>,
    pub subsystemclasses: Vec<Arc<SubsystemClass>>,
    pub shipais: Vec<Arc<ShipAI>>,
    pub shipflavors: Vec<Arc<ShipFlavor>>,
    pub squadronflavors: Vec<Arc<SquadronFlavor>>,
    pub unitclasses: Vec<UnitClass>,
    pub ships: RwLock<Vec<Arc<Ship>>>,
    pub squadrons: RwLock<Vec<Arc<Squadron>>>,
    pub unit_creation_counter: Arc<AtomicU64>,
    pub unit_death_counter: Arc<AtomicU64>,
    pub engagements: RwLock<Vec<EngagementRecord>>,
    pub global_salience: GlobalSalience,
    pub turn: Arc<AtomicU64>,
}

impl PartialEq for Root {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.nodeflavors == other.nodeflavors
            && self.nodes == other.nodes
            && self.clusters == other.clusters
            && self.edgeflavors == other.edgeflavors
            && self.edges == other.edges
            && self.neighbors == other.neighbors
            && self.factions == other.factions
            && self.wars == other.wars
            && self.resources == other.resources
            && self.hangarclasses == other.hangarclasses
            && self.hangar_counter.load(atomic::Ordering::Relaxed)
                == other.hangar_counter.load(atomic::Ordering::Relaxed)
            && self.engineclasses == other.engineclasses
            && self.repairerclasses == other.repairerclasses
            && self.factoryclasses == other.factoryclasses
            && self.shipyardclasses == other.shipyardclasses
            && self.shipais == other.shipais
            && self.shipflavors == other.shipflavors
            && self.squadronflavors == other.squadronflavors
            && self.unitclasses == other.unitclasses
            && self.ships.read().unwrap().clone() == other.ships.read().unwrap().clone()
            && self.squadrons.read().unwrap().clone() == other.squadrons.read().unwrap().clone()
            && self.unit_creation_counter.load(atomic::Ordering::Relaxed)
                == other.unit_creation_counter.load(atomic::Ordering::Relaxed)
            && self.unit_death_counter.load(atomic::Ordering::Relaxed)
                == other.unit_death_counter.load(atomic::Ordering::Relaxed)
            && self.engagements.read().unwrap().clone() == other.engagements.read().unwrap().clone()
            && self
                .global_salience
                .faction_salience
                .read()
                .unwrap()
                .clone()
                == other
                    .global_salience
                    .faction_salience
                    .read()
                    .unwrap()
                    .clone()
            && self
                .global_salience
                .resource_salience
                .read()
                .unwrap()
                .clone()
                == other
                    .global_salience
                    .resource_salience
                    .read()
                    .unwrap()
                    .clone()
            && self
                .global_salience
                .unitclass_salience
                .read()
                .unwrap()
                .clone()
                == other
                    .global_salience
                    .unitclass_salience
                    .read()
                    .unwrap()
                    .clone()
            && self.get_turn() == other.get_turn()
    }
}

impl Root {
    pub fn get_turn(&self) -> u64 {
        self.turn.load(atomic::Ordering::Relaxed)
    }
    //this is the method for creating a ship
    //duh
    pub fn create_ship(
        &self,
        class: Arc<ShipClass>,
        location: UnitLocation,
        faction: Arc<Faction>,
    ) -> Option<Arc<Ship>> {
        let mut ships_lock = self.ships.write().unwrap();
        let current_ship_num = ships_lock.len() as u64;
        let current_faction_ship_num = ships_lock
            .iter()
            .filter(|ship| ship.get_allegiance() == faction)
            .count() as u64;
        if current_ship_num < self.config.soft_ship_limit
            && current_faction_ship_num < faction.soft_ship_limit
        {
            //we call the shipclass instantiate method, and feed it the parameters it wants
            let new_ship = Arc::new(ShipClass::instantiate(
                class.clone(),
                location.clone(),
                faction,
                self,
            ));
            class.build_hangars(new_ship.clone(), &self.hangar_counter);
            ships_lock.push(new_ship.clone());
            location.insert_unit(
                &mut location.get_unit_container_write(),
                new_ship.get_unit(),
            );
            drop(ships_lock);
            Some(new_ship)
        } else {
            drop(ships_lock);
            None
        }
    }
    pub fn create_squadron(
        &self,
        class: Arc<SquadronClass>,
        location: UnitLocation,
        faction: Arc<Faction>,
    ) -> Arc<Squadron> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        let new_squadron = Arc::new(SquadronClass::instantiate(
            class.clone(),
            location.clone(),
            faction.clone(),
            self,
        ));

        let mut squadrons_lock = self.squadrons.write().unwrap();
        squadrons_lock.push(new_squadron.clone());
        location.insert_unit(
            &mut location.get_unit_container_write(),
            new_squadron.get_unit(),
        );
        drop(squadrons_lock);
        new_squadron
    }
    pub fn engagement_check(&self, node: Arc<Node>, actor: Arc<Faction>) -> Option<Arc<Faction>> {
        if node.mutables.read().unwrap().check_for_battles {
            let factions = node.get_node_factions(&self);
            if factions.iter().any(|f1| {
                factions.iter().any(|f2| {
                    self.wars
                        .contains(&(f1.clone().min(f2.clone()), f2.clone().max(f1.clone())))
                })
            }) {
                Some(actor)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn remove_dead(&self) {
        let disbanded = self
            .squadrons
            .read()
            .unwrap()
            .iter()
            .filter(|squadron| {
                if squadron.mutables.read().unwrap().ghost {
                    (squadron.get_strength(self.config.battle_scalars.avg_duration) as f32)
                        < (squadron.ideal_strength as f32 * squadron.class.disband_threshold)
                } else {
                    0 >= squadron.get_strength(self.config.battle_scalars.avg_duration)
                }
            })
            .cloned()
            .collect::<Vec<_>>();
        disbanded.iter().for_each(|squadron| {
            squadron.get_daughters().iter().for_each(|daughter| {
                assert!(daughter.transfer(UnitLocation::Node(squadron.get_mother_node())));
            });
        });
        let initial_dead: Vec<Unit> = self
            .ships
            .read()
            .unwrap()
            .iter()
            .map(|ship| ship.get_unit())
            .chain(
                self.squadrons
                    .read()
                    .unwrap()
                    .iter()
                    .map(|squadron| squadron.get_unit()),
            )
            .filter(|unit| !unit.is_alive())
            .collect();
        initial_dead.iter().for_each(|unit| {
            unit.clone().kill();
        });
        let all_dead: Vec<Unit> = initial_dead
            .iter()
            .flat_map(|unit| unit.get_daughters_recursive())
            .sorted()
            .dedup()
            .collect();
        assert!(all_dead.iter().all(|unit| !unit.is_alive()));
        all_dead
            .iter()
            .for_each(|dead_unit| match dead_unit.get_location() {
                UnitLocation::Node(node) => node
                    .unit_container
                    .write()
                    .unwrap()
                    .contents
                    .retain(|unit| unit.get_id() != dead_unit.get_id()),
                UnitLocation::Squadron(squadron) => squadron
                    .unit_container
                    .write()
                    .unwrap()
                    .contents
                    .retain(|unit| unit.get_id() != dead_unit.get_id()),
                UnitLocation::Hangar(hangar) => hangar
                    .unit_container
                    .write()
                    .unwrap()
                    .contents
                    .retain(|unit| unit.get_id() != dead_unit.get_id()),
            });
        all_dead
            .iter()
            .filter_map(|unit| unit.get_ship())
            .for_each(|ship| ship.mutables.write().unwrap().hangars = Vec::new());
        self.ships
            .write()
            .unwrap()
            .retain(|ship| !all_dead.contains(&ship.get_unit()));
        self.squadrons
            .write()
            .unwrap()
            .retain(|squadron| !all_dead.contains(&squadron.get_unit()));
        self.unit_death_counter
            .fetch_add(all_dead.len() as u64, atomic::Ordering::Relaxed);
    }
    pub fn process_turn(&mut self) {
        let turn_start = Instant::now();
        //increment turn counter
        let turn = self.turn.fetch_add(1, atomic::Ordering::Relaxed);
        println!("It is now turn {}.", turn);

        //reset node transaction flags
        self.nodes
            .iter()
            .for_each(|node| node.mutables.write().unwrap().resources_transacted = false);
        self.nodes
            .iter()
            .for_each(|node| node.mutables.write().unwrap().units_transacted = false);

        //reset all ships' engines
        self.ships
            .write()
            .unwrap()
            .iter()
            .for_each(|ship| ship.reset_movement());

        //run all ship repairers
        self.ships
            .write()
            .unwrap()
            .iter()
            .filter(|unit| unit.is_alive())
            .for_each(|ship| ship.repair(false));

        //process all factories
        self.nodes.iter().for_each(|n| n.process_factories());
        self.ships
            .write()
            .unwrap()
            .iter()
            .filter(|unit| unit.is_alive())
            .for_each(|ship| ship.process_factories());

        //process all shipyards
        self.nodes.iter().for_each(|n| n.process_shipyards());
        self.ships
            .write()
            .unwrap()
            .iter()
            .filter(|unit| unit.is_alive())
            .for_each(|ship| ship.process_shipyards());

        //plan ship creation
        let ship_plan_list: Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)> = self
            .nodes
            .iter()
            .map(|node| node.clone().plan_ships())
            .chain(
                self.ships
                    .write()
                    .unwrap()
                    .iter()
                    .filter(|ship| ship.is_alive())
                    .map(|ship| ship.plan_ships()),
            )
            .flatten()
            .collect();
        //create queued ships
        let n_new_ships = ship_plan_list
            .iter()
            .map(|(id, location, faction)| {
                self.create_ship(id.clone(), location.clone(), faction.clone())
            })
            .count();
        println!("Built {} new ships.", n_new_ships);

        let squadron_plan_list: Vec<(Arc<SquadronClass>, UnitLocation, Arc<Faction>)> = self
            .nodes
            .iter()
            .map(|node| node.clone().plan_squadrons(&self))
            .flatten()
            .collect();

        let n_new_squadrons = squadron_plan_list
            .iter()
            .map(|(id, location, faction)| {
                self.create_squadron(id.clone(), location.clone(), faction.clone());
                location.get_mother_node().transact_units(&self);
            })
            .count();
        println!("Created {} new squadrons.", n_new_squadrons);

        //propagate threat values
        //propagate saliences, create salience map
        //NOTE: I'm thinking of setting this up such that we use last turn's threat salience map as the degradation map for this turn's threat salience propagation.
        //That's what'll happen here with the way it is currently. Previous turn's values get used in calc faction salience, then get overwritten.
        //At least assuming the order of operations works like I'm expecting -- check that.
        //On turn 1, we'll want to run calc faction salience several times to get the values to settle.
        let salience_propagation_start = Instant::now();
        let gfs = self.calculate_global_faction_salience();
        *self.global_salience.faction_salience.write().unwrap() = gfs;
        let grs = self.calculate_global_resource_salience();
        *self.global_salience.resource_salience.write().unwrap() = grs;
        let gus = self.calculate_global_unitclass_salience();
        *self.global_salience.unitclass_salience.write().unwrap() = gus;
        let swem = self.calculate_strategic_weapon_effect_map();
        *self
            .global_salience
            .strategic_weapon_effect_map
            .write()
            .unwrap() = swem;
        let salience_propagation_finished = salience_propagation_start.elapsed();
        dbg!(salience_propagation_finished);

        //run operation management logic

        //move ships, one edge at a time
        //running battle checks and resource/unit balancing with each traversal
        let ship_moves_start = Instant::now();
        let ships = self.ships.read().unwrap().clone();
        ships
            .iter()
            .filter(|unit| unit.is_alive())
            .for_each(|ship| {
                let mut moving = true;
                while moving {
                    if ship.maneuver(&self).is_none() {
                        moving = false
                    }
                }
            });
        dbg!(ship_moves_start.elapsed());

        //move squadrons, one edge at a time
        //running battle checks and resource/unit balancing with each traversal

        //run defection logic

        //fire strategic weapons
        let mut rng = Hc128Rng::seed_from_u64(47);
        ships
            .iter()
            .filter(|unit| unit.is_alive())
            .for_each(|ship| {
                ship.mutables
                    .write()
                    .unwrap()
                    .strategic_weapons
                    .iter_mut()
                    .for_each(|weapon| {
                        weapon.fire(&self, ship.clone(), &mut rng);
                    })
            });

        //run diplomacy logic

        //run transact_resources/units on all nodes that have not already been affected
        self.nodes
            .iter()
            .filter(|node| !node.mutables.read().unwrap().resources_transacted)
            .for_each(|node| node.transact_resources(self));
        self.nodes
            .iter()
            .filter(|node| !node.mutables.read().unwrap().units_transacted)
            .for_each(|node| node.transact_units(self));

        self.remove_dead();

        //transmit root data to frontend

        let number_of_ships = &self.ships.read().unwrap().len();

        dbg!(number_of_ships);

        self.ships
            .read()
            .unwrap()
            .iter()
            .for_each(|ship| ship.check_location_coherency());

        self.squadrons
            .read()
            .unwrap()
            .iter()
            .for_each(|squadron| squadron.check_location_coherency());

        dbg!(turn_start.elapsed());
    }
}

#[cfg(test)]
mod test {
    use crate::internal::export::scale_from_threat;
    #[test]
    fn threat_scaling_test() {
        let inputs: Vec<f32> = vec![
            0.5_f32, -0.5_f32, 5_f32, -6_f32, -100_f32, 101_f32, 1042_f32, 5391_f32, -1632_f32,
            -9998_f32, -4141_f32, 43677_f32,
        ];

        for input in inputs {
            test_scale_from_threat(input);
        }
    }
    fn test_scale_from_threat(input: f32) {
        let scaled = scale_from_threat(input, 1000_f32);
        assert!(scaled < 1_f32);
        assert!(scaled > 0_f32);
        println!("{:05.1}\t{:.3}", input, scaled);
    }
    #[test]
    fn nav_calcs_test() {
        use ordered_float::NotNan;
        use rand::prelude::*;
        let mut rng = rand_hc::Hc128Rng::seed_from_u64(5803495084);
        let mut rdavs = Vec::new();
        let mut rsavs = Vec::new();
        for _ in 0..50 {
            let supply = rng.gen_range(0.0..10.0);
            let demand = rng.gen_range(0.0..10.0);
            let rdav = (demand - supply);
            let rsav = ((supply * demand) + ((supply - demand) * 5.0)) / 10.0;
            rdavs.push(NotNan::new(rdav).unwrap());
            rsavs.push(NotNan::new(rsav).unwrap());
            println!("Supply: {}; demand: {}", supply, demand);
            println!("Demand attraction value: {}", rdav);
            println!("Supply attraction value: {}", rsav);
            println!();
            println!();
        }
        println!(
            "Demand attraction min: {}; max: {}",
            rdavs.iter().min().unwrap(),
            rdavs.iter().max().unwrap()
        );
        println!(
            "Demand attraction average: {}",
            rdavs.iter().sum::<NotNan<f32>>() / rdavs.len() as f32
        );
        println!(
            "Supply attraction min: {}; max: {}",
            rsavs.iter().min().unwrap(),
            rsavs.iter().max().unwrap()
        );
        println!(
            "Supply attraction average: {}",
            rsavs.iter().sum::<NotNan<f32>>() / rsavs.len() as f32
        );
    }
}
