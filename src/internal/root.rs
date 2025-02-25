use crate::internal::engagement::EngagementRecord;
use crate::internal::faction::Faction;
use crate::internal::hangar::HangarClass;
use crate::internal::node::{EdgeFlavor, Locality, Node, NodeFlavor, System};
use crate::internal::resource::{
    EngineClass, FactoryClass, RepairerClass, Resource, ShipyardClass, StrategicWeaponClass,
};
use crate::internal::salience::GlobalSalience;
use crate::internal::unit::{
    Mobility, Ship, ShipAI, ShipClass, ShipFlavor, Squadron, SquadronClass, SquadronFlavor,
    SubsystemClass, Unit, UnitClassID, UnitLocation,
};
use itertools::Itertools;
use rand::prelude::*;
use rand_hc::Hc128Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub salience_scalars: SalienceScalars,
    pub entity_scalars: EntityScalars,
    pub battle_scalars: BattleScalars,
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
    System(Arc<System>),
    Unit(Unit),
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveTask {
    Reach,
    Kill,
    Protect,
    Capture,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Objective {
    pub target: ObjectiveTarget,
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
}

#[derive(Debug)]
pub struct Operation {
    pub visible_name: String,
    pub objectives: Vec<Objective>,
}

#[derive(Debug)]
pub struct Root {
    pub config: Config,
    pub nodeflavors: Vec<Arc<NodeFlavor>>,
    pub nodes: Vec<Arc<Node>>,
    pub systems: Vec<Arc<System>>,
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
    pub shipclasses: Vec<Arc<ShipClass>>,
    pub squadronclasses: Vec<Arc<SquadronClass>>,
    pub ships: RwLock<Vec<Arc<Ship>>>,
    pub squadrons: RwLock<Vec<Arc<Squadron>>>,
    pub unit_counter: Arc<AtomicU64>,
    pub engagements: RwLock<Vec<EngagementRecord>>,
    pub global_salience: GlobalSalience,
    pub turn: Arc<AtomicU64>,
}

impl PartialEq for Root {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.nodeflavors == other.nodeflavors
            && self.nodes == other.nodes
            && self.systems == other.systems
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
            && self.shipclasses == other.shipclasses
            && self.squadronclasses == other.squadronclasses
            && self.ships.read().unwrap().clone() == other.ships.read().unwrap().clone()
            && self.squadrons.read().unwrap().clone() == other.squadrons.read().unwrap().clone()
            && self.unit_counter.load(atomic::Ordering::Relaxed)
                == other.unit_counter.load(atomic::Ordering::Relaxed)
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
            && self.turn.load(atomic::Ordering::Relaxed)
                == other.turn.load(atomic::Ordering::Relaxed)
    }
}

impl Root {
    //this is the method for creating a ship
    //duh
    pub fn create_ship(
        &self,
        class: Arc<ShipClass>,
        location: UnitLocation,
        faction: Arc<Faction>,
    ) -> Arc<Ship> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        let new_ship = Arc::new(ShipClass::instantiate(
            class.clone(),
            location.clone(),
            faction,
            self,
        ));
        class.build_hangars(new_ship.clone(), &self.shipclasses, &self.hangar_counter);
        //NOTE: Is this thread-safe? There might be enough space in here
        //for something to go interact with the ship in root and fail to get the arc from location.
        self.ships.write().unwrap().push(new_ship.clone());
        location.insert_unit(
            &mut location.get_unit_container_write(),
            new_ship.get_unit(),
        );
        new_ship
    }
    pub fn create_squadron(
        &self,
        class: Arc<SquadronClass>,
        location: UnitLocation,
        faction: Arc<Faction>,
    ) -> Arc<Squadron> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        //let index_lock = RwLock::new(self.ships);
        let new_squadron = Arc::new(SquadronClass::instantiate(
            class.clone(),
            location.clone(),
            faction.clone(),
            self,
        ));

        class.ideal.iter().for_each(|(unitclass, _)| {
            let num = location
                .get_mother_node()
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .filter(|unit| &unit.get_allegiance() == &faction)
                .filter(|unit| {
                    &&UnitClassID::new_from_unitclass(&unit.get_unitclass()) == &unitclass
                })
                .count();
        });

        //NOTE: Is this thread-safe? There might be enough space in here
        //for something to go interact with the squadron in root and fail to get the arc from location.
        self.squadrons.write().unwrap().push(new_squadron.clone());
        location.insert_unit(
            &mut location.get_unit_container_write(),
            new_squadron.get_unit(),
        );
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
                ((squadron.get_strength(self.config.battle_scalars.avg_duration) as f32)
                    < (squadron.ideal_strength as f32 * squadron.class.disband_threshold))
                    && !squadron.mutables.read().unwrap().ghost
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
            .map(|node| node.clone().plan_ships(&self.shipclasses))
            .chain(
                self.ships
                    .write()
                    .unwrap()
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .map(|ship| ship.plan_ships(&self.shipclasses)),
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
        //At least assuming the order of operations works like I'm expecting -- check that with Amaryllis.
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
