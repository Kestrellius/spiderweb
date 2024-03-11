use itertools::Itertools;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_distr::*;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
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

#[derive(Debug, Clone)]
pub struct UnitContainer {
    pub contents: Vec<Unit>,
}

impl UnitContainer {
    pub fn new() -> UnitContainer {
        UnitContainer {
            contents: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFlavor {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
}

impl PartialEq for NodeFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for NodeFlavor {}

impl Ord for NodeFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for NodeFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for NodeFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeMut {
    pub visibility: bool,
    pub flavor: Arc<NodeFlavor>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub factories: Vec<Factory>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyards: Vec<Shipyard>,
    pub allegiance: Arc<Faction>, //faction that currently holds the node
    pub efficiency: f32, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub transact_resources: bool,
    pub transact_units: bool,
    pub check_for_battles: bool,
    pub resources_transacted: bool,
    pub units_transacted: bool,
}

#[derive(Debug)]
pub struct Node {
    pub id: usize,
    pub visible_name: String, //location name as shown to player
    pub position: [i64; 3], //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    pub description: String,
    pub environment: String, //name of the FRED environment to use for missions set in this node
    pub bitmap: Option<(String, f32)>, //name of the bitmap used to depict this node in other nodes' FRED environments, and a size factor
    pub mutables: RwLock<NodeMut>,
    pub unit_container: RwLock<UnitContainer>,
}

impl Node {
    fn get_strength(&self, faction: Arc<Faction>, time: u64) -> u64 {
        self.unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_strength(time))
            .sum()
    }
    pub fn get_node_forces(&self, root: &Root) -> HashMap<Arc<Faction>, Vec<Unit>> {
        root.factions
            .iter()
            .map(|faction| {
                let units: Vec<Unit> = self
                    .unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .filter(|unit| unit.get_allegiance() == *faction)
                    .cloned()
                    .collect();
                (faction.clone(), units)
            })
            .filter(|(_, units)| units.len() > 0)
            .collect()
    }
    pub fn get_node_factions(&self, root: &Root) -> Vec<Arc<Faction>> {
        root.factions
            .iter()
            .filter(|faction| {
                !self
                    .unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .filter(|unit| unit.get_allegiance() == **faction)
                    .collect::<Vec<_>>()
                    .is_empty()
            })
            .cloned()
            .collect()
    }
    pub fn get_node_faction_reinforcements(
        &self,
        destination: Arc<Node>,
        factionid: Arc<Faction>,
        root: &Root,
    ) -> Vec<Unit> {
        let top_level_units: Vec<Unit> = self
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .filter(|unit| unit.get_allegiance() == factionid)
            .filter(|unit| {
                unit.destinations_check(root, &vec![destination.clone()])
                    .is_some()
            })
            .cloned()
            .collect();
        let daughter_units: Vec<Unit> = top_level_units
            .iter()
            .map(|unit| unit.get_traversal_checked_daughters(root, destination.clone()))
            .flatten()
            .collect();
        top_level_units
            .into_iter()
            .chain(daughter_units.into_iter())
            .collect()
    }
    pub fn get_node_faction_forces(
        &self,
        faction: Arc<Faction>,
    ) -> (Vec<Arc<Squadron>>, Vec<Arc<Ship>>) {
        let ships: Vec<Arc<Ship>> = self
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.mutables.read().unwrap().allegiance == faction)
            .collect();
        let squadrons: Vec<Arc<Squadron>> = self
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .filter_map(|unit| unit.get_squadron())
            .filter(|squadron| squadron.mutables.read().unwrap().allegiance == faction)
            .collect();
        (squadrons, ships)
    }
    pub fn get_system(&self, root: &Root) -> Option<Arc<System>> {
        let system = root.systems.iter().find(|system| {
            system
                .nodes
                .iter()
                .find(|sys_node| sys_node.id == self.id)
                .is_some()
        });
        match system {
            Some(sys) => Some(sys.clone()),
            None => None,
        }
    }
    pub fn is_in_system(&self, system: Arc<System>) -> bool {
        system
            .nodes
            .iter()
            .find(|sys_node| sys_node.id == self.id)
            .is_some()
    }
    pub fn process_factories(&self) {
        let efficiency = self.mutables.read().unwrap().efficiency;
        self.mutables
            .write()
            .unwrap()
            .factories
            .iter_mut()
            .for_each(|f| f.process(efficiency));
    }
    pub fn process_shipyards(&self) {
        let efficiency = self.mutables.read().unwrap().efficiency;
        self.mutables
            .write()
            .unwrap()
            .shipyards
            .iter_mut()
            .for_each(|sy| sy.process(efficiency));
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.visible_name == other.visible_name
            && self.position == other.position
            && self.description == other.description
            && self.environment == other.environment
            && self.bitmap == other.bitmap
            && self.mutables.read().unwrap().clone() == other.mutables.read().unwrap().clone()
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

trait Locality {
    fn get_distance(&self, rhs: Arc<Node>) -> u64;
    fn get_nodes_in_range(
        &self,
        root: &Root,
        range: u64,
        forbidden_nodeflavors: &Vec<Arc<NodeFlavor>>,
        forbidden_edgeflavors: &Vec<Arc<EdgeFlavor>>,
    ) -> Vec<Arc<Node>>;
    fn plan_ships(
        &self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)>;
    fn plan_squadrons(&self, root: &Root) -> Vec<(Arc<SquadronClass>, UnitLocation, Arc<Faction>)>;
    fn transact_resources(&self, root: &Root);
    fn transact_units(&self, root: &Root);
}

impl Locality for Arc<Node> {
    fn get_distance(&self, rhs: Arc<Node>) -> u64 {
        let self_pos = self.position;
        let rhs_pos = rhs.position;
        (((self_pos[0] - rhs_pos[0]) + (self_pos[1] - rhs_pos[1]) + (self_pos[2] - rhs_pos[2]))
            as f32)
            .sqrt() as u64
    }
    fn get_nodes_in_range(
        &self,
        root: &Root,
        range: u64,
        forbidden_nodeflavors: &Vec<Arc<NodeFlavor>>,
        forbidden_edgeflavors: &Vec<Arc<EdgeFlavor>>,
    ) -> Vec<Arc<Node>> {
        fn inner_recursion(
            self_node: Arc<Node>,
            root: &Root,
            range: u64,
            forbidden_nodeflavors: &Vec<Arc<NodeFlavor>>,
            forbidden_edgeflavors: &Vec<Arc<EdgeFlavor>>,
            touched: &mut HashMap<Arc<Node>, u64>,
        ) {
            if !forbidden_nodeflavors.contains(&self_node.mutables.read().unwrap().flavor) {
                touched.insert(self_node.clone(), range);
            }
            if range > 0 {
                let valid_neighbors: Vec<_> = root
                    .neighbors
                    .get(&self_node)
                    .unwrap()
                    .iter()
                    .cloned()
                    .filter(|neighbor| {
                        touched
                            .get(neighbor)
                            .map(|range_left_when_touched| *range_left_when_touched <= range)
                            .unwrap_or(true)
                    })
                    .filter(|neighbor| {
                        !forbidden_nodeflavors.contains(&neighbor.mutables.read().unwrap().flavor)
                    })
                    .filter(|neighbor| {
                        !forbidden_edgeflavors.contains(
                            root.edges
                                .get(&(
                                    (&self_node).min(neighbor).clone(),
                                    (neighbor).max(&self_node).clone(),
                                ))
                                .unwrap(),
                        )
                    })
                    .collect();
                valid_neighbors.iter().for_each(|neighbor| {
                    assert!(touched.insert(neighbor.clone(), range).is_none());
                });
                valid_neighbors.iter().for_each(|_neighbor| {
                    inner_recursion(
                        self_node.clone(),
                        root,
                        range,
                        forbidden_nodeflavors,
                        forbidden_edgeflavors,
                        touched,
                    )
                });
            }
        }
        let mut touched = HashMap::new();
        inner_recursion(
            self.clone(),
            root,
            range,
            forbidden_nodeflavors,
            forbidden_edgeflavors,
            &mut touched,
        );
        touched.into_keys().collect::<Vec<_>>()
    }
    fn plan_ships(
        &self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)> {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        let allegiance = mutables.allegiance.clone();
        mutables
            .shipyards
            .iter_mut()
            .map(|shipyard| {
                let ship_plans = shipyard.plan_ships(efficiency, shipclasses);
                //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                ship_plans
                    .iter()
                    .map(|ship_plan| {
                        (
                            ship_plan.clone(),
                            UnitLocation::Node(self.clone()),
                            allegiance.clone(),
                        )
                    })
                    // <^>>(
                    .collect::<Vec<_>>()
            })
            //we flatten the collection of vecs corresponding to individual shipyards, because we just want to create all the ships and don't care who created them
            .flatten()
            .collect::<Vec<_>>()
    }
    fn plan_squadrons(&self, root: &Root) -> Vec<(Arc<SquadronClass>, UnitLocation, Arc<Faction>)> {
        root.factions
            .iter()
            .map(|faction| {
                let mut avail_units: HashMap<UnitClassID, u64> =
                    root.shipclasses
                        .iter()
                        .map(|shipclass| ShipClass::get_unitclass(shipclass.clone()))
                        .chain(root.squadronclasses.iter().map(|squadronclass| {
                            SquadronClass::get_unitclass(squadronclass.clone())
                        }))
                        .map(|unitclass| {
                            (
                                UnitClassID::new_from_unitclass(&unitclass),
                                self.unit_container
                                    .read()
                                    .unwrap()
                                    .contents
                                    .iter()
                                    .filter(|unit| unit.is_alive())
                                    .filter(|unit| &unit.get_allegiance() == faction)
                                    .filter(|unit| &unit.get_unitclass() == &unitclass)
                                    .count() as u64,
                            )
                        })
                        .collect();
                root.squadronclasses
                    .iter()
                    .map(move |squadronclass| {
                        let mut adding_squadrons = true;
                        let mut added_squadrons = Vec::new();
                        while adding_squadrons {
                            let avail_nums: HashMap<UnitClassID, u64> = squadronclass
                                .ideal
                                .iter()
                                .map(|(idealclassid, num)| {
                                    (
                                        *idealclassid,
                                        *avail_units.get(idealclassid).unwrap().min(num),
                                    )
                                })
                                .collect();
                            let avail_strength: f32 = avail_nums
                                .iter()
                                .map(|(idealclassid, num)| {
                                    (idealclassid.get_unitclass(root).get_ideal_strength(root)
                                        * num) as f32
                                })
                                .sum();
                            if avail_strength / squadronclass.get_ideal_strength(root) as f32
                                > squadronclass.creation_threshold
                            {
                                squadronclass.ideal.iter().for_each(|(idealclassid, num)| {
                                    let avail_num = avail_units.get_mut(idealclassid).unwrap();
                                    *avail_num = avail_num.saturating_sub(*num);
                                    added_squadrons.push((
                                        squadronclass.clone(),
                                        UnitLocation::Node(self.clone()),
                                        faction.clone(),
                                    ));
                                })
                            } else {
                                adding_squadrons = false
                            }
                        }
                        added_squadrons
                    })
                    .flatten()
            })
            .flatten()
            .collect()
    }
    fn transact_resources(&self, root: &Root) {
        let mut node_mut = self.mutables.write().unwrap();
        let node_container = self.unit_container.read().unwrap();
        root.factions.iter().for_each(|faction| {
            let node_is_owned = faction.id == node_mut.allegiance.id;
            let all_relevant_ships: Vec<Arc<Ship>> = node_container
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .filter(|unit| unit.get_allegiance().id == faction.id)
                .map(|unit| unit.get_daughters_recursive())
                .flatten()
                .filter_map(|unit| unit.get_ship())
                .collect();
            let mut all_relevant_ships_mut_lock: Vec<RwLockWriteGuard<ShipMut>> =
                all_relevant_ships
                    .iter()
                    .map(|ship| ship.mutables.write().unwrap())
                    .filter(|ship_mut| {
                        !(ship_mut.engines.iter().all(|e| e.inputs.is_empty())
                            && ship_mut.repairers.iter().all(|r| r.inputs.is_empty())
                            && ship_mut
                                .strategic_weapons
                                .iter()
                                .all(|w| w.inputs.is_empty())
                            && ship_mut.factories.iter().all(|f| f.inputs.is_empty())
                            && ship_mut.factories.iter().all(|f| f.outputs.is_empty())
                            && ship_mut.shipyards.iter().all(|s| s.inputs.is_empty()))
                    })
                    .collect::<Vec<_>>();

            root.resources.iter().for_each(|resource| {
                let unipotent_stockpiles_supply_demand: Vec<(u64, u64)> =
                    all_relevant_ships_mut_lock
                        .iter()
                        .map(|ship_mut| {
                            ship_mut
                                .engines
                                .iter()
                                .flat_map(|engine| {
                                    engine
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                })
                                .chain(ship_mut.repairers.iter().flat_map(|repairer| {
                                    repairer
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                                .chain(ship_mut.strategic_weapons.iter().flat_map(|weapon| {
                                    weapon
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                                .chain(ship_mut.factories.iter().flat_map(|factory| {
                                    factory
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                                .chain(ship_mut.factories.iter().flat_map(|factory| {
                                    factory
                                        .outputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                                .chain(ship_mut.shipyards.iter().flat_map(|shipyard| {
                                    shipyard
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                        })
                        .flatten()
                        .chain(Some(()).iter().filter(|_| node_is_owned).flat_map(|_| {
                            node_mut
                                .factories
                                .iter()
                                .flat_map(|factory| {
                                    factory
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                })
                                .chain(node_mut.factories.iter().flat_map(|factory| {
                                    factory
                                        .outputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                                .chain(node_mut.shipyards.iter().flat_map(|shipyard| {
                                    shipyard
                                        .inputs
                                        .iter()
                                        .filter(|usp| usp.resource_type.id == resource.id)
                                        .map(|usp| (usp.contents, usp.target))
                                }))
                        }))
                        .collect();

                let total_pluripotent_naive_demand = all_relevant_ships_mut_lock
                    .iter()
                    .map(|ship_mut| {
                        ship_mut
                            .stockpiles
                            .iter()
                            .map(|sp| {
                                sp.target.saturating_sub(
                                    sp.get_fullness()
                                        .saturating_sub(sp.get_resource_num(resource.clone())),
                                )
                            })
                            .sum::<u64>()
                    })
                    .sum::<u64>() as f32;

                let external_demand = (root.global_salience.resource_salience.read().unwrap()
                    [faction.id][resource.id][self.id][0]
                    - (unipotent_stockpiles_supply_demand
                        .iter()
                        .map(|(supply, _)| *supply as f32)
                        .sum::<f32>()
                        + total_pluripotent_naive_demand))
                    .clamp(0.0, f32::MAX);

                let pluripotent_stockpiles_supply_demand: Vec<Vec<(u64, u64)>> =
                    all_relevant_ships_mut_lock
                        .iter()
                        .map(|ship_mut| {
                            ship_mut
                                .stockpiles
                                .iter()
                                .map(|ppsp| {
                                    (
                                        ppsp.get_resource_supply(resource.clone()),
                                        ppsp.get_pluripotent_transaction_resource_demand(
                                            resource.clone(),
                                            external_demand,
                                            ship_mut.get_speed_factor(
                                                root.config.entity_scalars.avg_speed,
                                                root.turn.load(atomic::Ordering::Relaxed),
                                            ),
                                            total_pluripotent_naive_demand,
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();

                let supply_total = unipotent_stockpiles_supply_demand
                    .iter()
                    .map(|(supply, _)| *supply)
                    .sum::<u64>()
                    + pluripotent_stockpiles_supply_demand
                        .iter()
                        .map(|x| x.iter().map(|(supply, _)| *supply).sum::<u64>())
                        .sum::<u64>();

                let demand_total = unipotent_stockpiles_supply_demand
                    .iter()
                    .map(|(_, demand)| *demand)
                    .sum::<u64>()
                    + pluripotent_stockpiles_supply_demand
                        .iter()
                        .map(|x| x.iter().map(|(_, demand)| *demand).sum::<u64>())
                        .sum::<u64>();

                let supply_demand_ratio = supply_total as f32 / demand_total as f32;

                let mut transfer_stockpile = UnipotentStockpile {
                    visibility: false,
                    resource_type: resource.clone(),
                    contents: 0,
                    rate: 0,
                    target: 0,
                    capacity: u64::MAX,
                    propagates: false,
                };

                all_relevant_ships_mut_lock.iter_mut().for_each(|ship_mut| {
                    ship_mut.engines.iter_mut().for_each(|engine| {
                        engine
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.repairers.iter_mut().for_each(|repairer| {
                        repairer
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.strategic_weapons.iter_mut().for_each(|weapon| {
                        weapon
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .outputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.shipyards.iter_mut().for_each(|shipyard| {
                        shipyard
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                });

                if node_is_owned {
                    node_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            });
                    });
                    node_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .outputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            });
                    });
                    node_mut.shipyards.iter_mut().for_each(|shipyard| {
                        shipyard
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    usp.contents.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        usp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            });
                    });
                }

                all_relevant_ships_mut_lock.iter_mut().enumerate().for_each(
                    |(ship_i, ship_mut)| {
                        ship_mut
                            .stockpiles
                            .iter_mut()
                            .enumerate()
                            .for_each(|(sp_i, ppsp)| {
                                let (sp_supply, sp_demand) =
                                    pluripotent_stockpiles_supply_demand[ship_i][sp_i];
                                let proper_quantity =
                                    (sp_demand as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity = sp_supply.saturating_sub(proper_quantity);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        ppsp,
                                        &mut transfer_stockpile,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    },
                );

                //transfer_stockpile now contains all the resources that we're transferring
                //so we begin transferring them back out, into the stockpiles that need them

                all_relevant_ships_mut_lock.iter_mut().for_each(|ship_mut| {
                    ship_mut.engines.iter_mut().for_each(|engine| {
                        engine
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.repairers.iter_mut().for_each(|repairer| {
                        repairer
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.strategic_weapons.iter_mut().for_each(|weapon| {
                        weapon
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .outputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                    ship_mut.shipyards.iter_mut().for_each(|shipyard| {
                        shipyard
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    });
                });

                if node_is_owned {
                    node_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            });
                    });
                    node_mut.factories.iter_mut().for_each(|factory| {
                        factory
                            .outputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            });
                    });
                    node_mut.shipyards.iter_mut().for_each(|shipyard| {
                        shipyard
                            .inputs
                            .iter_mut()
                            .filter(|usp| usp.resource_type.id == resource.id)
                            .for_each(|usp| {
                                let proper_quantity =
                                    (usp.target as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity =
                                    proper_quantity.saturating_sub(usp.contents);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        usp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            });
                    });
                }

                all_relevant_ships_mut_lock.iter_mut().enumerate().for_each(
                    |(ship_i, ship_mut)| {
                        ship_mut
                            .stockpiles
                            .iter_mut()
                            .enumerate()
                            .for_each(|(sp_i, ppsp)| {
                                let (sp_supply, sp_demand) =
                                    pluripotent_stockpiles_supply_demand[ship_i][sp_i];
                                let proper_quantity =
                                    (sp_demand as f32 * supply_demand_ratio) as u64;
                                let transfer_quantity = proper_quantity.saturating_sub(sp_supply);
                                if transfer_quantity > 0 {
                                    Stockpileness::transfer(
                                        &mut transfer_stockpile,
                                        ppsp,
                                        resource.clone(),
                                        transfer_quantity,
                                    );
                                }
                            })
                    },
                );
            });
        });

        node_mut.resources_transacted = true;

        //acquire write lock on all ships in node

        //collect all unipotent and pluripotent stockpiles in node into two lists

        //for each resource:

        //gather all real supply in node

        //gather all demand in node, calculating unipotent stockpile demand normally
        //and calculating pluripotent stockpile demand as demand from outside node * ship's speed/average speed * stockpile's capacity as fraction of all pluripotent stockpiles' capacity, clamped by stockpile's target

        //calculate supply to demand ratio: supply/demand
        //this will let us determine what fraction of the supply should be given to an entity that has a given amount of demand

        //iterate over all unipotent stockpiles
        //for each, get the stockpile's supply and demand; then
        //while lhs contents < lhs supply:
        //iterate over unipotent stockpiles (except lhs); for each, determine transfer amount, which is:
        //lhs demand saturating-sub (rhs supply * supply/demand ratio), clamped by rhs contents saturating-sub (rhs supply * supply/demand ratio)
        //transfer transfer amount from rhs to lhs
        //iterate over pluripotent stockpiles (as rhs) and do same
        //once both iterations are done, break
        //iterate over pluripotent stockpiles (as lhs) and do same
    }
    fn transact_units(&self, root: &Root) {
        let mut node_mut = self.mutables.write().unwrap();
        let mut node_container = self.unit_container.write().unwrap();
        root.factions.iter().for_each(|faction| {
            let all_units = node_container
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .filter(|unit| unit.get_allegiance().id == faction.id)
                .flat_map(|unit| unit.get_daughters_recursive())
                .collect::<Vec<_>>();

            let all_squadrons: Vec<Arc<Squadron>> = all_units
                .iter()
                .filter_map(|unit| unit.get_squadron())
                .collect();

            let all_squadrons_indexed: HashMap<u64, Arc<Squadron>> = all_squadrons
                .iter()
                .map(|squadron| (squadron.id, squadron.clone()))
                .collect();

            let mut all_squadrons_mut_lock: HashMap<u64, RwLockWriteGuard<SquadronMut>> =
                all_squadrons_indexed
                    .iter()
                    .map(|(index, squadron)| (*index, squadron.mutables.write().unwrap()))
                    .collect();

            let mut all_squadrons_containers_lock: HashMap<u64, RwLockWriteGuard<UnitContainer>> =
                all_squadrons
                    .iter()
                    .map(|squadron| (squadron.id, squadron.unit_container.write().unwrap()))
                    .collect();

            let squadron_container_is_not_empty_map: HashMap<u64, bool> =
                all_squadrons_containers_lock
                    .iter()
                    .map(|(id, container)| (*id, !container.contents.is_empty()))
                    .collect();

            let all_ships: Vec<Arc<Ship>> = all_units
                .iter()
                .filter_map(|unit| unit.get_ship())
                .collect();

            let mut all_ships_mut_lock: HashMap<u64, RwLockWriteGuard<ShipMut>> = all_ships
                .iter()
                .map(|ship| (ship.id, ship.mutables.write().unwrap()))
                .collect();

            let all_hangars_indexed_with_speed: HashMap<u64, (Arc<Hangar>, f32)> =
                all_ships_mut_lock
                    .iter()
                    .map(|(_, ship_mut)| {
                        let speed_factor = ship_mut.get_speed_factor(
                            root.config.entity_scalars.avg_speed,
                            root.turn.load(atomic::Ordering::Relaxed),
                        );
                        ship_mut
                            .hangars
                            .iter()
                            .map(|hangar| (hangar.id, (hangar.clone(), speed_factor)))
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .collect();

            let mut all_hangars_containers_lock: HashMap<u64, RwLockWriteGuard<UnitContainer>> =
                all_hangars_indexed_with_speed
                    .iter()
                    .map(|(_, (hangar, _))| (hangar.id, hangar.unit_container.write().unwrap()))
                    .collect();

            let unitclasses = root
                .shipclasses
                .iter()
                .map(|shipclass| ShipClass::get_unitclass(shipclass.clone()))
                .chain(
                    root.squadronclasses
                        .iter()
                        .map(|squadronclass| SquadronClass::get_unitclass(squadronclass.clone())),
                )
                .collect::<Vec<_>>();

            unitclasses.iter().for_each(|unitclass| {
                let node_supply: u64 = all_units
                    .iter()
                    .filter(|unit| match unit {
                        Unit::Ship(ship) => {
                            all_ships_mut_lock.get(&ship.id).unwrap().location
                                == UnitLocation::Node(self.clone())
                        }
                        Unit::Squadron(squadron) => {
                            all_squadrons_mut_lock.get(&squadron.id).unwrap().location
                                == UnitLocation::Node(self.clone())
                        }
                    })
                    .filter(|unit| unit.get_unitclass() == unitclass.clone())
                    .map(|unit| {
                        unit.get_real_volume_locked(
                            &all_squadrons_containers_lock,
                            &all_ships_mut_lock,
                        )
                    })
                    .sum();

                let squadrons_supply: HashMap<u64, u64> = all_squadrons_indexed
                    .iter()
                    .map(|(i, squadron)| {
                        (
                            *i,
                            squadron.get_unitclass_supply_local(
                                all_squadrons_containers_lock.get(&i).unwrap(),
                                unitclass.clone(),
                                &all_ships_mut_lock,
                                &all_squadrons_containers_lock,
                            ),
                        )
                    })
                    .collect();

                let squadrons_demand: HashMap<u64, u64> = all_squadrons_indexed
                    .iter()
                    .map(|(i, squadron)| {
                        (
                            *i,
                            squadron.get_unitclass_demand_local(
                                all_squadrons_containers_lock.get(&i).unwrap(),
                                unitclass.clone(),
                                &all_ships_mut_lock,
                                &all_squadrons_containers_lock,
                            ),
                        )
                    })
                    .collect();

                let hangars_supply: HashMap<u64, u64> = all_hangars_indexed_with_speed
                    .iter()
                    .map(|(i, (hangar, _))| {
                        (
                            *i,
                            hangar.get_unitclass_supply_local(
                                all_hangars_containers_lock.get(&i).unwrap(),
                                unitclass.clone(),
                                &all_ships_mut_lock,
                                &all_squadrons_containers_lock,
                            ),
                        )
                    })
                    .collect();

                let non_transport_hangars_demand: HashMap<u64, u64> =
                    all_hangars_indexed_with_speed
                        .iter()
                        .filter(|(_, (hangar, _))| !hangar.class.transport)
                        .map(|(i, (hangar, _))| {
                            (
                                *i,
                                hangar.get_unitclass_demand_local(
                                    all_hangars_containers_lock.get(&i).unwrap(),
                                    unitclass.clone(),
                                    &all_ships_mut_lock,
                                    &all_squadrons_containers_lock,
                                ),
                            )
                        })
                        .collect();

                let total_transport_naive_demand = all_hangars_indexed_with_speed
                    .iter()
                    .filter(|(_, (hangar, _))| hangar.class.transport)
                    .map(|(i, (hangar, _))| {
                        hangar.get_unitclass_demand_local(
                            all_hangars_containers_lock.get(&i).unwrap(),
                            unitclass.clone(),
                            &all_ships_mut_lock,
                            &all_squadrons_containers_lock,
                        )
                    })
                    .sum::<u64>();

                let external_demand: f32 =
                    (root.global_salience.unitclass_salience.read().unwrap()[faction.id]
                        [unitclass.get_id()][self.id][0]
                        - (squadrons_demand.values().sum::<u64>()
                            + non_transport_hangars_demand.values().sum::<u64>()
                            + total_transport_naive_demand) as f32)
                        .clamp(0.0, f32::MAX);

                let transport_hangars_demand: HashMap<u64, u64> = all_hangars_indexed_with_speed
                    .iter()
                    .filter(|(_, (hangar, _))| hangar.class.transport)
                    .map(|(i, (hangar, speed_factor))| {
                        (
                            *i,
                            hangar.get_transport_transaction_unitclass_demand(
                                all_hangars_containers_lock.get(&i).unwrap(),
                                unitclass.clone(),
                                &all_ships_mut_lock,
                                external_demand,
                                *speed_factor,
                                total_transport_naive_demand as f32,
                            ),
                        )
                    })
                    .collect();

                let supply_total = node_supply
                    + squadrons_supply
                        .iter()
                        .map(|(_, supply)| *supply)
                        .sum::<u64>()
                    + hangars_supply
                        .iter()
                        .map(|(_, supply)| *supply)
                        .sum::<u64>();

                let demand_total = squadrons_demand
                    .iter()
                    .map(|(_, demand)| *demand)
                    .sum::<u64>()
                    + non_transport_hangars_demand
                        .iter()
                        .map(|(_, demand)| *demand)
                        .sum::<u64>()
                    + transport_hangars_demand
                        .iter()
                        .map(|(_, demand)| *demand)
                        .sum::<u64>();

                let supply_demand_ratio = supply_total as f32 / demand_total as f32;

                all_squadrons_indexed.iter().for_each(|(index, squadron)| {
                    let container = all_squadrons_containers_lock.get(index).unwrap();
                    let proper_quantity: u64 =
                        (*squadrons_demand.get(index).unwrap() as f32 * supply_demand_ratio) as u64;
                    let mut quantity_transferred: u64 = 0;
                    let unit_volumes: HashMap<u64, u64> = container
                        .contents
                        .iter()
                        .filter(|unit| {
                            unit.is_alive_locked(
                                &all_ships_mut_lock,
                                &all_squadrons_mut_lock,
                                &squadron_container_is_not_empty_map,
                            )
                        })
                        .filter(|unit| &unit.get_unitclass() == unitclass)
                        .map(|unit| {
                            (
                                unit.get_id(),
                                unit.get_real_volume_locked(
                                    &all_squadrons_containers_lock,
                                    &all_ships_mut_lock,
                                ),
                            )
                        })
                        .collect();
                    let initial_quantity: u64 = unit_volumes.values().sum();
                    let mut container_mut = all_squadrons_containers_lock.get_mut(index).unwrap();
                    let relevant_container_contents = container_mut
                        .contents
                        .iter()
                        .filter(|unit| {
                            unit.is_alive_locked(
                                &all_ships_mut_lock,
                                &all_squadrons_mut_lock,
                                &squadron_container_is_not_empty_map,
                            )
                        })
                        .filter(|unit| &unit.get_unitclass() == unitclass)
                        .cloned()
                        .collect::<Vec<_>>();
                    for contained_unit in relevant_container_contents {
                        let unit_volume = unit_volumes.get(&contained_unit.get_id()).unwrap();
                        if initial_quantity
                            .saturating_sub(quantity_transferred)
                            .saturating_sub(unit_volume / 2)
                            < proper_quantity
                        {
                            break;
                        };
                        let unit_location = UnitLocation::Squadron(squadron.clone());
                        contained_unit.transfer_locked(
                            unit_location,
                            &mut container_mut,
                            UnitLocation::Node(self.clone()),
                            &mut node_container,
                            &mut all_ships_mut_lock,
                            &mut all_squadrons_mut_lock,
                            &squadron_container_is_not_empty_map,
                        );
                        quantity_transferred += unit_volume;
                    }
                });

                all_hangars_indexed_with_speed
                    .iter()
                    .for_each(|(index, (hangar, _))| {
                        let container = all_hangars_containers_lock.get(index).unwrap();
                        let proper_quantity: u64 = if hangar.class.transport {
                            (*transport_hangars_demand.get(index).unwrap() as f32
                                * supply_demand_ratio) as u64
                        } else {
                            (*non_transport_hangars_demand.get(index).unwrap() as f32
                                * supply_demand_ratio) as u64
                        };
                        let mut quantity_transferred: u64 = 0;
                        let unit_volumes: HashMap<u64, u64> = container
                            .contents
                            .iter()
                            .filter(|unit| {
                                unit.is_alive_locked(
                                    &all_ships_mut_lock,
                                    &all_squadrons_mut_lock,
                                    &squadron_container_is_not_empty_map,
                                )
                            })
                            .filter(|unit| &unit.get_unitclass() == unitclass)
                            .map(|unit| {
                                (
                                    unit.get_id(),
                                    unit.get_real_volume_locked(
                                        &all_squadrons_containers_lock,
                                        &all_ships_mut_lock,
                                    ),
                                )
                            })
                            .collect();
                        let initial_quantity: u64 = unit_volumes.values().sum();
                        let mut container_mut = all_hangars_containers_lock.get_mut(index).unwrap();
                        let relevant_container_contents = container_mut
                            .contents
                            .iter()
                            .filter(|unit| {
                                unit.is_alive_locked(
                                    &all_ships_mut_lock,
                                    &all_squadrons_mut_lock,
                                    &squadron_container_is_not_empty_map,
                                )
                            })
                            .filter(|unit| &unit.get_unitclass() == unitclass)
                            .cloned()
                            .collect::<Vec<_>>();
                        for contained_unit in relevant_container_contents {
                            let unit_volume = unit_volumes.get(&contained_unit.get_id()).unwrap();
                            if initial_quantity
                                .saturating_sub(quantity_transferred)
                                .saturating_sub(unit_volume / 2)
                                < proper_quantity
                            {
                                break;
                            };
                            let unit_location = UnitLocation::Hangar(hangar.clone());
                            contained_unit.transfer_locked(
                                unit_location,
                                &mut container_mut,
                                UnitLocation::Node(self.clone()),
                                &mut node_container,
                                &mut all_ships_mut_lock,
                                &mut all_squadrons_mut_lock,
                                &squadron_container_is_not_empty_map,
                            );
                            quantity_transferred += unit_volume;
                        }
                    });

                all_squadrons_indexed.iter().for_each(|(index, squadron)| {
                    let squadron_container = all_squadrons_containers_lock.get(index).unwrap();
                    let squadron_initial_quantity: u64 = squadron_container
                        .contents
                        .iter()
                        .filter(|unit| {
                            unit.is_alive_locked(
                                &all_ships_mut_lock,
                                &all_squadrons_mut_lock,
                                &squadron_container_is_not_empty_map,
                            )
                        })
                        .filter(|unit| &unit.get_unitclass() == unitclass)
                        .map(|unit| {
                            unit.get_real_volume_locked(
                                &all_squadrons_containers_lock,
                                &all_ships_mut_lock,
                            )
                        })
                        .sum();
                    let proper_quantity: u64 =
                        (*squadrons_demand.get(index).unwrap() as f32 * supply_demand_ratio) as u64;
                    let mut quantity_transferred: u64 = 0;
                    let unit_volumes: HashMap<u64, u64> = all_units
                        .iter()
                        .filter(|unit| match unit {
                            Unit::Ship(ship) => {
                                all_ships_mut_lock.get(&ship.id).unwrap().location
                                    == UnitLocation::Node(self.clone())
                            }
                            Unit::Squadron(squadron) => {
                                all_squadrons_mut_lock.get(&squadron.id).unwrap().location
                                    == UnitLocation::Node(self.clone())
                            }
                        })
                        .filter(|unit| &unit.get_unitclass() == unitclass)
                        .map(|unit| {
                            (
                                unit.get_id(),
                                unit.get_real_volume_locked(
                                    &all_squadrons_containers_lock,
                                    &all_ships_mut_lock,
                                ),
                            )
                        })
                        .collect();
                    let mut squadron_container_mut =
                        all_squadrons_containers_lock.get_mut(index).unwrap();
                    let relevant_node_container_contents = all_units
                        .iter()
                        .filter(|unit| match unit {
                            Unit::Ship(ship) => {
                                all_ships_mut_lock.get(&ship.id).unwrap().location
                                    == UnitLocation::Node(self.clone())
                            }
                            Unit::Squadron(squadron) => {
                                all_squadrons_mut_lock.get(&squadron.id).unwrap().location
                                    == UnitLocation::Node(self.clone())
                            }
                        })
                        .filter(|unit| &unit.get_unitclass() == unitclass)
                        .filter(|unit| {
                            all_ships_mut_lock.get(&unit.get_id()).is_some()
                                || all_squadrons_mut_lock.get(&unit.get_id()).is_some()
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    for contained_unit in relevant_node_container_contents {
                        let unit_volume = unit_volumes.get(&contained_unit.get_id()).unwrap();
                        if squadron_initial_quantity + (quantity_transferred) + (unit_volume / 2)
                            > proper_quantity
                        {
                            break;
                        };
                        let unit_location = UnitLocation::Node(self.clone());
                        contained_unit.transfer_locked(
                            unit_location,
                            &mut node_container,
                            UnitLocation::Squadron(squadron.clone()),
                            &mut squadron_container_mut,
                            &mut all_ships_mut_lock,
                            &mut all_squadrons_mut_lock,
                            &squadron_container_is_not_empty_map,
                        );
                        quantity_transferred += unit_volume;
                    }
                });

                all_hangars_indexed_with_speed
                    .iter()
                    .for_each(|(index, (hangar, _))| {
                        let hangar_container = all_hangars_containers_lock.get(index).unwrap();
                        let hangar_initial_quantity: u64 = hangar_container
                            .contents
                            .iter()
                            .filter(|unit| {
                                unit.is_alive_locked(
                                    &all_ships_mut_lock,
                                    &all_squadrons_mut_lock,
                                    &squadron_container_is_not_empty_map,
                                )
                            })
                            .filter(|unit| &unit.get_unitclass() == unitclass)
                            .map(|unit| {
                                unit.get_real_volume_locked(
                                    &all_squadrons_containers_lock,
                                    &all_ships_mut_lock,
                                )
                            })
                            .sum();
                        let proper_quantity: u64 = if hangar.class.transport {
                            (*transport_hangars_demand.get(index).unwrap() as f32
                                * supply_demand_ratio) as u64
                        } else {
                            (*non_transport_hangars_demand.get(index).unwrap() as f32
                                * supply_demand_ratio) as u64
                        };
                        let mut quantity_transferred: u64 = 0;
                        let unit_volumes: HashMap<u64, u64> = all_units
                            .iter()
                            .filter(|unit| match unit {
                                Unit::Ship(ship) => {
                                    all_ships_mut_lock.get(&ship.id).unwrap().location
                                        == UnitLocation::Node(self.clone())
                                }
                                Unit::Squadron(squadron) => {
                                    all_squadrons_mut_lock.get(&squadron.id).unwrap().location
                                        == UnitLocation::Node(self.clone())
                                }
                            })
                            .filter(|unit| &unit.get_unitclass() == unitclass)
                            .map(|unit| {
                                (
                                    unit.get_id(),
                                    unit.get_real_volume_locked(
                                        &all_squadrons_containers_lock,
                                        &all_ships_mut_lock,
                                    ),
                                )
                            })
                            .collect();
                        let mut hangar_container_mut =
                            all_hangars_containers_lock.get_mut(index).unwrap();
                        let relevant_node_container_contents = all_units
                            .iter()
                            .filter(|unit| match unit {
                                Unit::Ship(ship) => {
                                    all_ships_mut_lock.get(&ship.id).unwrap().location
                                        == UnitLocation::Node(self.clone())
                                }
                                Unit::Squadron(squadron) => {
                                    all_squadrons_mut_lock.get(&squadron.id).unwrap().location
                                        == UnitLocation::Node(self.clone())
                                }
                            })
                            .filter(|unit| &unit.get_unitclass() == unitclass)
                            .filter(|unit| {
                                all_ships_mut_lock.get(&unit.get_id()).is_some()
                                    || all_squadrons_mut_lock.get(&unit.get_id()).is_some()
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        for contained_unit in relevant_node_container_contents {
                            let unit_volume = unit_volumes.get(&contained_unit.get_id()).unwrap();
                            if hangar_initial_quantity + (quantity_transferred) + (unit_volume / 2)
                                > proper_quantity
                            {
                                break;
                            };
                            let unit_location = UnitLocation::Node(self.clone());
                            contained_unit.transfer_locked(
                                unit_location,
                                &mut node_container,
                                UnitLocation::Hangar(hangar.clone()),
                                &mut hangar_container_mut,
                                &mut all_ships_mut_lock,
                                &mut all_squadrons_mut_lock,
                                &squadron_container_is_not_empty_map,
                            );
                            quantity_transferred += unit_volume;
                        }
                    });
            });
        });
        node_mut.units_transacted = true;
    }
}

#[derive(Debug, Clone)]
pub struct System {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub nodes: Vec<Arc<Node>>,
}

impl PartialEq for System {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for System {}

impl Ord for System {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for System {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for System {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFlavor {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub propagates: bool,
}

impl PartialEq for EdgeFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for EdgeFlavor {}

impl Ord for EdgeFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for EdgeFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for EdgeFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edges {
    hyperlinks: HashSet<(Arc<Node>, Arc<Node>, Arc<EdgeFlavor>)>, //list of links between nodes
    neighbormap: HashMap<Arc<Node>, Vec<Arc<Node>>>, //NOTE: investigate. Map of which nodes belong to which systems, for purposes of generating all-to-all links
}

impl Edges {
    //this creates an edge between two nodes, and adds each node to the other's neighbor map
    fn insert(&mut self, a: Arc<Node>, b: Arc<Node>, flavor: Arc<EdgeFlavor>) {
        assert_ne!(a.clone(), b.clone());
        self.hyperlinks
            .insert((a.clone().max(b.clone()), a.clone().min(b.clone()), flavor));
        self.neighbormap
            .entry(a.clone())
            .or_insert_with(|| Vec::new())
            .push(b.clone());
        self.neighbormap
            .entry(b)
            .or_insert_with(|| Vec::new())
            .push(a);
    }
    /*fn insert_with_distance(&mut self, root: &mut Root, a: Arc<Node>, b: Arc<Node>, distance: u64) {
        for i in 0..=distance {
            let p = root.create_node(0, None, None, null, etc);
            self.insert(a, p)
        }
    }*/
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct FactionID {
    pub index: usize,
}

impl FactionID {
    pub fn new_from_index(index: usize) -> Self {
        FactionID { index: index }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Faction {
    pub id: usize,
    pub visible_name: String, //faction name as shown to player
    pub description: String,
    pub visibility: bool,
    pub propagates: bool,
    pub efficiency_default: f32, //starting value for production facility efficiency
    pub efficiency_target: f32, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    pub efficiency_delta: f32,  //rate at which efficiency changes
    pub battle_scalar: f32,
    pub value_mult: f32, //how valuable the AI considers one point of this faction's threat to be
    pub volume_strength_ratio: f32, //faction's multiplier for resource/unitclass supply points when comparing to threat values for faction demand calcs
    pub relations: HashMap<FactionID, f32>,
}

impl PartialEq for Faction {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Faction {}

impl Ord for Faction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Faction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Faction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub propagates: bool,
    pub unit_vol: u64, //how much volume (in e.g. cubic meters) one unit of this resource takes up; this is intended only for player-facing use
    pub value_mult: f32, //how valuable the AI considers one unit of this resource to be
}

impl PartialEq for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Resource {}

impl Ord for Resource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Resource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Resource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub trait Stockpileness {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64>;
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64;
    fn get_capacity(&self) -> u64;
    fn get_fullness(&self) -> u64;
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>>;
    fn get_resource_supply(&self, resourceid: Arc<Resource>) -> u64;
    fn get_resource_demand(&self, resourceid: Arc<Resource>) -> u64;
    fn get_pluripotent_transaction_resource_demand(
        &self,
        resource: Arc<Resource>,
        external_demand: f32,
        speed_factor: f32,
        total_target: f32,
    ) -> u64;
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64;
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64;
    fn transfer<S: Stockpileness>(&mut self, rhs: &mut S, cargo: Arc<Resource>, quantity: u64) {
        let extracted = self.remove(cargo.clone(), quantity);
        let remainder = rhs.insert(cargo.clone(), extracted);
        self.insert(cargo, remainder);
    }
}

//this is a horrible incomprehensible nightmare that Amaryllis put me through for some reason
//okay, so, a year later, what this actually does is that it takes two individual stockpiles and allows them to function together as a single stockpile
impl<A: Stockpileness, B: Stockpileness> Stockpileness for (A, B) {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.0
            .collate_contents()
            .iter()
            .chain(self.1.collate_contents().iter())
            .map(|(resource, value)| (resource.clone(), *value))
            .collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        self.0.get_resource_num(cargo.clone()) + self.1.get_resource_num(cargo.clone())
    }
    fn get_capacity(&self) -> u64 {
        self.0.get_capacity() + self.1.get_capacity()
    }
    fn get_fullness(&self) -> u64 {
        self.0.get_fullness() + self.1.get_fullness()
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        //self.0
        //    .get_allowed()
        //    .iter()
        //    .chain(self.1.get_allowed().iter())
        //    .collect()
        Some(Vec::new())
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        self.0.get_resource_supply(resource.clone()) + self.1.get_resource_supply(resource.clone())
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        self.0.get_resource_demand(resource.clone()) + self.1.get_resource_demand(resource.clone())
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        resource: Arc<Resource>,
        external_demand: f32,
        speed_factor: f32,
        total_target: f32,
    ) -> u64 {
        self.0.get_pluripotent_transaction_resource_demand(
            resource.clone(),
            external_demand,
            speed_factor,
            total_target,
        ) + self.1.get_pluripotent_transaction_resource_demand(
            resource.clone(),
            external_demand,
            speed_factor,
            total_target,
        )
    }
    fn insert(&mut self, _cargo: Arc<Resource>, quantity: u64) -> u64 {
        quantity
    }
    fn remove(&mut self, _cargo: Arc<Resource>, _quantity: u64) -> u64 {
        0
    }
}

//a unipotent resource stockpile can contain only one type of resource
//however, the quantity of resource specified in the rate field may be added to or removed from the stockpile under various circumstances,
//such as once every turn, depending on how it's used
#[derive(Debug, Clone, PartialEq)]
pub struct UnipotentStockpile {
    pub visibility: bool,
    pub resource_type: Arc<Resource>,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl UnipotentStockpile {
    fn input_is_sufficient(&self) -> bool {
        self.contents >= self.rate
    }
    //this is the logic to determine whether a unipotent stockpile should be active, dormant, or stalled
    fn output_state(&self) -> OutputState {
        //NOTE: Dormancy is dummied out for now
        //if self.contents >= self.target {
        //    OutputState::Dormant
        //} else
        //NOTE: This is not perfect because we don't have easy access to efficiency here,
        //so we don't know exactly how much will be added to the stockpile the next time it's incremented.
        //However, if capacity is exceeded, the contents will be capped out properly in output_process,
        //and the next time output_state is run, it will detect as stalled.
        if self.contents + self.rate >= self.capacity {
            OutputState::Stalled
        } else {
            OutputState::Active
        }
    }
    fn input_process(&mut self) {
        let subtracted: Option<u64> = self.contents.checked_sub(self.rate);
        if let Some(new) = subtracted {
            self.contents = new;
        } else {
            panic!("Factory input stockpile is too low.")
        }
    }
    fn output_process(&mut self, efficiency: f32) {
        self.contents += (self.rate as f32 * efficiency) as u64;
        if self.contents >= self.capacity {
            self.contents = self.capacity
        }
    }
}

impl Stockpileness for UnipotentStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((self.resource_type.clone(), self.contents)).collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resource_type {
            self.contents
        } else {
            0
        }
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self) -> u64 {
        self.contents
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        Some(vec![self.resource_type.clone()])
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resource_type {
            (self.contents).saturating_sub(self.target)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resource_type {
            self.target.saturating_sub(self.contents)
        } else {
            0
        }
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        _resource: Arc<Resource>,
        _external_demand: f32,
        _speed_factor: f32,
        _total_target: f32,
    ) -> u64 {
        0
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resource_type {
            let old_contents = self.contents;
            let remainder = quantity.saturating_sub(self.capacity - old_contents);
            self.contents += quantity - remainder;
            assert!(self.contents <= self.capacity);
            assert_eq!(self.contents + remainder, old_contents + quantity);
            remainder
        } else {
            quantity
        }
    }
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resource_type {
            let _old_contents = self.contents;
            let remainder = quantity.saturating_sub(self.contents);
            self.contents -= quantity - remainder;
            quantity - remainder
        } else {
            0
        }
    }
}

//a pluripotent stockpile can contain any number of different resources and ships
//however, it has no constant rate of increase or decrease; things may only be added or removed manually
#[derive(Debug, Clone, PartialEq)]
pub struct PluripotentStockpile {
    pub visibility: bool,
    pub contents: HashMap<Arc<Resource>, u64>,
    pub allowed: Option<Vec<Arc<Resource>>>,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl Stockpileness for PluripotentStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.contents.clone()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        *self.contents.get(&cargo).unwrap_or(&0)
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self) -> u64 {
        self.contents.iter().map(|(_, value)| value).sum::<u64>()
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        self.allowed.clone()
    }
    //unlike other places, here in pluripotent stockpiles we don't take target into account when calculating supply
    //thus, items in pluripotent stockpiles always emit supply, even if the stockpile still wants more
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        let sum = self.get_resource_num(resource.clone());
        sum
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        if self
            .allowed
            .as_ref()
            .map(|allowed| allowed.contains(&resource))
            .unwrap_or(true)
        {
            self.target.saturating_sub(self.get_fullness())
        } else {
            0
        }
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        resource: Arc<Resource>,
        external_demand: f32,
        speed_factor: f32,
        total_naive_demand: f32,
    ) -> u64 {
        if self
            .allowed
            .as_ref()
            .map(|allowed| allowed.contains(&resource))
            .unwrap_or(true)
        {
            let space_left_without_resource = self.target.saturating_sub(
                self.get_fullness()
                    .saturating_sub(self.get_resource_num(resource.clone())),
            );
            ((external_demand
                * speed_factor
                * (space_left_without_resource as f32 / total_naive_demand)) as u64)
                .clamp(0, space_left_without_resource) as u64
        } else {
            0
        }
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if self
            .allowed
            .clone()
            .map(|allowed| allowed.contains(&cargo))
            .unwrap_or(true)
        {
            let old_fullness = self.get_fullness();
            let remainder = quantity.saturating_sub(self.capacity - old_fullness);
            *self.contents.get_mut(&cargo).unwrap() += quantity - remainder;
            assert!(self.get_fullness() <= self.capacity);
            assert_eq!(self.get_fullness() + remainder, old_fullness + quantity);
            remainder
        } else {
            quantity
        }
    }
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if self
            .allowed
            .clone()
            .map(|allowed| allowed.contains(&cargo))
            .unwrap_or(true)
        {
            let old_contents = self.contents.get(&cargo).unwrap();
            let remainder = quantity.saturating_sub(*old_contents);
            *self.contents.get_mut(&cargo).unwrap() -= quantity - remainder;
            quantity - remainder
        } else {
            0
        }
    }
}

//a given shared stockpile type has its contents shared between all instances of itself; it does not produce any salience propagation
//previously this was an atomic I64 and we don't remember why, but an atomic U64 seems better and hopefully it doesn't subtly break something
#[derive(Debug, Clone)]
pub struct SharedStockpile {
    pub resource_type: Arc<Resource>,
    pub contents: Arc<AtomicU64>,
    pub rate: u64,
    pub capacity: u64,
}

impl Stockpileness for SharedStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((
            self.resource_type.clone(),
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resource_type {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self) -> u64 {
        self.contents.load(atomic::Ordering::SeqCst)
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        Some(vec![self.resource_type.clone()])
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resource_type {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, _resourceid: Arc<Resource>) -> u64 {
        0
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        _resource: Arc<Resource>,
        _external_demand: f32,
        _speed_factor: f32,
        _total_target: f32,
    ) -> u64 {
        0
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resource_type {
            let old_contents = self.contents.load(atomic::Ordering::SeqCst);
            let remainder = quantity.saturating_sub(self.capacity - old_contents);
            self.contents
                .fetch_add(quantity - remainder, atomic::Ordering::SeqCst);
            assert!(self.contents.load(atomic::Ordering::SeqCst) <= self.capacity);
            assert_eq!(
                self.contents.load(atomic::Ordering::SeqCst) + remainder,
                old_contents + quantity
            );
            remainder
        } else {
            quantity
        }
    }
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resource_type {
            let old_contents = self.contents.load(atomic::Ordering::SeqCst);
            let remainder = quantity.saturating_sub(old_contents);
            self.contents
                .fetch_sub(quantity - remainder, atomic::Ordering::SeqCst);
            quantity - remainder
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HangarClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub capacity: u64,                     //total volume the hangar can hold
    pub target: u64, //volume the hangar wants to hold; this is usually equal to capacity
    pub allowed: Option<Vec<UnitClassID>>, //which shipclasses this hangar can hold
    pub ideal: HashMap<UnitClassID, u64>, //how many of each ship type the hangar wants
    pub sub_target_supply_scalar: f32, //multiplier used for supply generated by non-ideal units under the target limit; should be below one
    pub non_ideal_demand_scalar: f32, //multiplier used for demand generated for non-ideal unitclasses; should be below one
    pub transport: bool, //whether hangar exists primarily to transport ships to a destination, rather than launch ships into combat or utility roles
    pub launch_volume: u64, //how much volume the hangar can launch at one time in battle
    pub launch_interval: u64, //time between launches in battle
    pub propagates: bool, //whether or not hangar generates saliences
}

impl PartialEq for HangarClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for HangarClass {}

impl Ord for HangarClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for HangarClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for HangarClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl HangarClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.ideal
            .iter()
            .map(|(unitclass, v)| {
                let ideal_strength = match unitclass {
                    UnitClassID::ShipClass(sc) => {
                        root.shipclasses[sc.index].get_ideal_strength(root)
                    }
                    UnitClassID::SquadronClass(fc) => {
                        root.squadronclasses[fc.index].get_ideal_strength(root)
                    }
                };
                ideal_strength * v
            })
            .sum()
    }
    pub fn instantiate(
        class: Arc<Self>,
        mother: Arc<Ship>,
        _shipclasses: &Vec<Arc<ShipClass>>,
        counter: &Arc<AtomicU64>,
    ) -> Hangar {
        let index = counter.fetch_add(1, atomic::Ordering::Relaxed);
        Hangar {
            id: index,
            class: class.clone(),
            mother: mother.clone(),
            mutables: RwLock::new(HangarMut {
                visibility: class.visibility,
            }),
            unit_container: RwLock::new(UnitContainer::new()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HangarMut {
    pub visibility: bool,
}

#[derive(Debug)]
pub struct Hangar {
    pub id: u64,
    pub class: Arc<HangarClass>,
    pub mother: Arc<Ship>,
    pub mutables: RwLock<HangarMut>,
    pub unit_container: RwLock<UnitContainer>,
}

impl Hangar {
    pub fn get_strength(&self, time: u64) -> u64 {
        let contents_strength = self
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .map(|unit| unit.get_strength(time))
            .sum::<u64>() as f32;
        let contents_vol = self
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .map(|unit| unit.get_real_volume())
            .sum::<u64>() as f32;
        //we calculate how much of its complement the hangar can launch during a battle a certain number of seconds long
        let launch_mod = ((contents_vol / self.class.launch_volume as f32)
            * (time as f32 / self.class.launch_interval as f32))
            .clamp(0.0, 1.0);
        (contents_strength * launch_mod) as u64
    }
    pub fn get_fullness(&self) -> u64 {
        self.unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .map(|unit| unit.get_real_volume())
            .sum()
    }
    pub fn get_unitclass_num_recursive(&self, unitclass: UnitClass) -> u64 {
        let unitclass_id = UnitClassID::new_from_unitclass(&unitclass);
        if self
            .class
            .allowed
            .as_ref()
            .map(|allowed_vec| allowed_vec.contains(&unitclass_id))
            .unwrap_or(true)
        {
            self.unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .map(|daughter| daughter.get_unitclass_num(unitclass.clone()))
                .sum::<u64>()
        } else {
            0
        }
    }
    pub fn get_unitclass_supply_recursive(&self, unitclass: UnitClass) -> u64 {
        let unitclass_id = UnitClassID::new_from_unitclass(&unitclass);
        if self
            .class
            .allowed
            .as_ref()
            .map(|allowed_vec| allowed_vec.contains(&unitclass_id))
            .unwrap_or(true)
        {
            let daughter_supply = self
                .unit_container
                .read()
                .unwrap()
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .map(|daughter| daughter.get_unitclass_supply_recursive(unitclass.clone()))
                .sum::<u64>();
            let ideal_volume = self
                .class
                .ideal
                .get(&UnitClassID::new_from_unitclass(&unitclass))
                .unwrap_or(&0)
                * unitclass.get_ideal_volume();
            let non_ideal_volume = daughter_supply.saturating_sub(ideal_volume);
            let excess_volume = self.get_fullness().saturating_sub(self.class.target);
            let over_target_supply = (excess_volume).min(non_ideal_volume);
            let under_target_supply = ((non_ideal_volume.saturating_sub(over_target_supply)) as f32
                * self.class.sub_target_supply_scalar) as u64;
            over_target_supply + under_target_supply
        } else {
            0
        }
    }
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
    pub fn get_unitclass_demand_recursive(&self, unitclass: UnitClass) -> u64 {
        let unitclass_id = UnitClassID::new_from_unitclass(&unitclass);
        if self
            .class
            .allowed
            .as_ref()
            .map(|allowed_vec| allowed_vec.contains(&unitclass_id))
            .unwrap_or(true)
        {
            let unit_container = self.unit_container.read().unwrap();
            let daughter_volume = unit_container
                .contents
                .iter()
                .filter(|unit| unit.is_alive())
                .filter(|unit| &unit.get_unitclass() == &unitclass)
                .map(|unit| unit.get_real_volume())
                .sum::<u64>();
            let ideal_volume = self
                .class
                .ideal
                .get(&UnitClassID::new_from_unitclass(&unitclass))
                .unwrap_or(&0)
                * unitclass.get_ideal_volume();
            let ideal_demand = ideal_volume.saturating_sub(daughter_volume);
            let non_ideal_demand = (self
                .class
                .target
                .saturating_sub(self.get_fullness())
                .saturating_sub(ideal_demand) as f32
                * self.class.non_ideal_demand_scalar) as u64;
            ideal_demand
                + non_ideal_demand
                + unit_container
                    .contents
                    .iter()
                    .filter(|unit| unit.is_alive())
                    .map(|daughter| daughter.get_unitclass_demand_recursive(unitclass.clone()))
                    .sum::<u64>()
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
            let daughter_volume = unit_container
                .contents
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
            let ideal_demand = ideal_volume.saturating_sub(daughter_volume);
            let non_ideal_demand = (self
                .class
                .target
                .saturating_sub(
                    unit_container
                        .contents
                        .iter()
                        .filter(|unit| {
                            unit.get_ship()
                                .map(|ship| ships_mut_lock.get(&ship.id).unwrap().hull.get() > 0)
                                .unwrap_or(true)
                        })
                        .map(|unit| {
                            unit.get_real_volume_locked(squadrons_containers_lock, ships_mut_lock)
                        })
                        .sum::<u64>(),
                )
                .saturating_sub(ideal_demand) as f32
                * self.class.non_ideal_demand_scalar) as u64;
            ideal_demand + non_ideal_demand
        } else {
            0
        }
    }
    pub fn get_transport_transaction_unitclass_demand(
        &self,
        unit_container: &RwLockWriteGuard<UnitContainer>,
        unitclass: UnitClass,
        ships_mut_lock: &HashMap<u64, RwLockWriteGuard<ShipMut>>,
        external_demand: f32,
        speed_factor: f32,
        total_naive_demand: f32,
    ) -> u64 {
        let unitclass_id = UnitClassID::new_from_unitclass(&unitclass);
        if self
            .class
            .allowed
            .as_ref()
            .map(|allowed_vec| allowed_vec.contains(&unitclass_id))
            .unwrap_or(true)
        {
            let ideal_volume =
                self.class.ideal.get(&unitclass_id).unwrap_or(&0) * unitclass.get_ideal_volume();
            let space_left_without_unitclass = self.class.target.saturating_sub(
                unit_container
                    .contents
                    .iter()
                    .filter(|unit| {
                        unit.get_ship()
                            .map(|ship| ships_mut_lock.get(&ship.id).unwrap().hull.get() > 0)
                            .unwrap_or(true)
                    })
                    .filter(|unit| unit.get_unitclass().get_id() != unitclass.get_id())
                    .map(|unit| unit.get_real_volume())
                    .sum(),
            );
            let non_ideal_volume = space_left_without_unitclass.saturating_sub(ideal_volume);
            let self_demand = ideal_volume
                + (non_ideal_volume as f32 * self.class.non_ideal_demand_scalar) as u64;
            ((external_demand * speed_factor * (self_demand as f32 / total_naive_demand)) as u64)
                .clamp(0, self_demand)
        } else {
            0
        }
    }
}

impl PartialEq for Hangar {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Hangar {}

impl Ord for Hangar {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Hangar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Hangar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub trait ResourceProcess {
    fn get_state(&self) -> FactoryState;
    fn get_resource_supply_total(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64;
    fn get_target_total(&self) -> u64;
}

#[derive(Debug, Clone)]
pub struct EngineClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub base_health: Option<u64>,
    pub toughness_scalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl EngineClass {
    pub fn instantiate(class: Arc<Self>) -> Engine {
        Engine {
            class: class.clone(),
            visibility: class.visibility,
            health: class.base_health,
            inputs: class.inputs.clone(),
            forbidden_nodeflavors: class.forbidden_nodeflavors.clone(),
            forbidden_edgeflavors: class.forbidden_edgeflavors.clone(),
            last_move_turn: 0,
        }
    }
}

impl PartialEq for EngineClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for EngineClass {}

impl Ord for EngineClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for EngineClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for EngineClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Engine {
    pub class: Arc<EngineClass>,
    pub visibility: bool,
    pub health: Option<u64>,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub last_move_turn: u64,
}

impl Engine {
    fn check_engine(
        &self,
        root: &Root,
        location: Arc<Node>,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<(Vec<Arc<Node>>, u64)> {
        if (self.health != Some(0))
            && ((root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn)
                > self.class.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            let viable: Vec<Arc<Node>> = destinations
                .iter()
                .filter(|destination| {
                    self.nav_check(root, location.clone(), destination.clone().clone())
                })
                .cloned()
                .collect();
            if !viable.is_empty() {
                Some((viable, self.class.speed))
            } else {
                None
            }
        } else {
            None
        }
    }
    fn check_engine_movement_only(&self, turn: u64) -> bool {
        if (self.health != Some(0))
            && ((turn - self.last_move_turn) > self.class.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            true
        } else {
            false
        }
    }
    //this is run once per turn for a given engine; it checks to see if the engine has enough resources to run this turn and whether it's off cooldown
    //then consumes stockpile resources, and returns the engine's speed
    //we'll need to reset movement_left to max at the start of the turn
    fn process_engine(
        &mut self,
        root: &Root,
        location: Arc<Node>,
        destination: Arc<Node>,
    ) -> Option<u64> {
        if (self.health != Some(0))
            && ((root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn)
                > self.class.cooldown)
            && (self.get_state() == FactoryState::Active)
            && (self.nav_check(root, location, destination))
        {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.last_move_turn = root.turn.load(atomic::Ordering::Relaxed);
            Some(self.class.speed)
        } else {
            None
        }
    }
    fn nav_check(&self, root: &Root, location: Arc<Node>, destination: Arc<Node>) -> bool {
        !self
            .forbidden_nodeflavors
            .contains(&destination.mutables.read().unwrap().flavor)
            && root
                .edges
                .get(&(
                    location.clone().min(destination.clone()),
                    destination.max(location),
                ))
                .map(|edge| !self.forbidden_edgeflavors.contains(edge))
                .unwrap_or(false)
    }
    fn get_moves_left(&self, movement_left: u64) -> u64 {
        (self
            .inputs
            .iter()
            .map(|sp| sp.contents / sp.rate)
            .min()
            .unwrap_or(0))
        .min(movement_left / self.class.speed)
    }
}

impl ResourceProcess for Engine {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
}

#[derive(Debug, Clone)]
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
    pub subsystem_repair_points: i64,
    pub subsystem_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerClass {
    pub fn instantiate(class: Arc<Self>) -> Repairer {
        Repairer {
            class: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
        }
    }
}

impl PartialEq for RepairerClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RepairerClass {}

impl Ord for RepairerClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for RepairerClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for RepairerClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Repairer {
    pub class: Arc<RepairerClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
}

impl Repairer {
    fn process(&mut self) {
        self.inputs
            .iter_mut()
            .for_each(|input| input.input_process());
    }
}

impl ResourceProcess for Repairer {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategicWeaponClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the weapon won't fire into nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the weapon won't fire across edges of these flavors
    pub damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage done by a single shot, first by points and then by factor
    pub engine_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to engine done by a single shot, first by points and then by factor
    pub strategic_weapon_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to strategic weapon done by a single shot, first by points and then by factor
    pub accuracy: f32, //divided by target's strategicweaponevasionscalar to get hit probability as a fraction of 1.0
    pub range: u64,    //how many edges away the weapon can reach
    pub shots: (u64, u64), //lower and upper bounds for maximum number of shots the weapon fires each time it's activated
    pub targets_enemies: bool,
    pub targets_allies: bool,
    pub targets_neutrals: bool,
    pub target_relations_lower_bound: Option<f32>,
    pub target_relations_upper_bound: Option<f32>,
    pub target_priorities_class: HashMap<ShipClassID, f32>, //how strongly weapon will prioritize ships of each class; classes absent from list will default to 1.0
    pub target_priorities_flavor: HashMap<Arc<ShipFlavor>, f32>, //how strongly weapon will prioritize ships of each flavor; flavors absent from list will default to 1.0
}

impl StrategicWeaponClass {
    pub fn instantiate(class: Arc<Self>) -> StrategicWeapon {
        StrategicWeapon {
            class: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
        }
    }
}

impl Eq for StrategicWeaponClass {}

impl Ord for StrategicWeaponClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for StrategicWeaponClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for StrategicWeaponClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategicWeapon {
    pub class: Arc<StrategicWeaponClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
}

impl StrategicWeapon {
    fn targets_faction(
        &self,
        root: &Root,
        allegiance: &Arc<Faction>,
        faction: &Arc<Faction>,
    ) -> bool {
        let enemies = root
            .factions
            .iter()
            .cloned()
            .filter(|rhs_faction| {
                root.wars.contains(&(
                    (rhs_faction).min(&allegiance).clone(),
                    (allegiance).max(&rhs_faction).clone(),
                ))
            })
            .collect::<Vec<_>>();
        (self.class.targets_enemies && enemies.contains(&faction))
            || (self.class.targets_allies && faction == allegiance)
            || (self.class.targets_neutrals
                && !(faction == allegiance || enemies.contains(&faction)))
            || (self
                .class
                .target_relations_upper_bound
                .map(|val| {
                    allegiance
                        .relations
                        .get(&FactionID::new_from_index(faction.id))
                        .unwrap()
                        < &val
                })
                .unwrap_or(false))
            || (self
                .class
                .target_relations_upper_bound
                .map(|val| {
                    allegiance
                        .relations
                        .get(&FactionID::new_from_index(faction.id))
                        .unwrap()
                        > &val
                })
                .unwrap_or(false))
    }
    fn fire<R: Rng>(
        &mut self,
        root: &Root,
        mother: Arc<Ship>,
        mut rng: &mut R,
    ) -> HashMap<Unit, UnitStatus> {
        let allegiance = &mother.get_allegiance();
        let location = mother.get_mother_node();

        let target_nodes = location.get_nodes_in_range(
            root,
            self.class.range,
            &self.class.forbidden_nodeflavors,
            &self.class.forbidden_edgeflavors,
        );

        let target_factions = root
            .factions
            .iter()
            .cloned()
            .filter(|faction| self.targets_faction(root, &allegiance, faction))
            .collect::<Vec<_>>();

        let targets: Vec<(Arc<Ship>, f32)> = target_nodes
            .iter()
            .map(|node| {
                node.unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .map(|unit| unit.get_undocked_daughters())
                    .flatten()
                    .filter(|ship| ship.is_alive())
                    .filter(|ship| {
                        target_factions.contains(&&ship.mutables.read().unwrap().allegiance)
                            && ship.id != mother.id
                    })
                    .map(|ship| {
                        (
                            ship.clone(),
                            self.class
                                .target_priorities_class
                                .get(&ShipClassID::new_from_index(ship.class.id))
                                .unwrap_or(&0.0)
                                + self
                                    .class
                                    .target_priorities_flavor
                                    .get(&ship.class.shipflavor)
                                    .unwrap_or(&0.0),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        let shots_fired: usize = rng.gen_range(self.class.shots.0..self.class.shots.1) as usize;

        (0..shots_fired).for_each(|_| self.inputs.iter_mut().for_each(|sp| sp.input_process()));

        let hit_ships: Vec<Arc<Ship>> = (0..shots_fired)
            .map(|_| {
                targets
                    .choose_weighted(&mut rng, |(_, weight)| weight.clone())
                    .unwrap()
            })
            .map(|(ship, _)| ship.clone())
            .filter(|target| {
                (self.class.accuracy / target.class.strategic_weapon_evasion_scalar) > 1.0
            })
            .collect();

        let hit_ships_status: HashMap<Unit, UnitStatus> = hit_ships
            .iter()
            .map(|hit_ship| {
                let status = UnitStatus {
                    location: Some(hit_ship.mutables.read().unwrap().location.clone()),
                    damage: ((rng
                        .gen_range(self.class.damage.0 .0 as f32..self.class.damage.0 .1 as f32)
                        / hit_ship.class.toughness_scalar)
                        + ((rng.gen_range(
                            self.class.damage.1 .0 as f32..self.class.damage.1 .1 as f32,
                        ) / hit_ship.class.toughness_scalar)
                            * hit_ship.class.base_hull as f32)) as i64,
                    engine_damage: hit_ship
                        .mutables
                        .read()
                        .unwrap()
                        .engines
                        .iter()
                        .filter(|e| e.health.is_some())
                        .map(|e| {
                            ((rng.gen_range(
                                self.class.engine_damage.0 .0 as f32
                                    ..self.class.engine_damage.0 .1 as f32,
                            ) / e.class.toughness_scalar)
                                + ((rng.gen_range(
                                    self.class.engine_damage.1 .0 as f32
                                        ..self.class.engine_damage.1 .1 as f32,
                                ) / hit_ship.class.toughness_scalar)
                                    * e.class.base_health.unwrap() as f32))
                                as i64
                        })
                        .collect(),
                    subsystem_damage: hit_ship
                        .mutables
                        .read()
                        .unwrap()
                        .subsystems
                        .iter()
                        .filter(|s| s.health.is_some())
                        .map(|s| {
                            ((rng.gen_range(
                                self.class.strategic_weapon_damage.0 .0 as f32
                                    ..self.class.strategic_weapon_damage.0 .1 as f32,
                            ) / s.class.toughness_scalar)
                                + ((rng.gen_range(
                                    self.class.strategic_weapon_damage.1 .0 as f32
                                        ..self.class.strategic_weapon_damage.1 .1 as f32,
                                ) / hit_ship.class.toughness_scalar)
                                    * s.class.base_health.unwrap() as f32))
                                as i64
                        })
                        .collect(),
                };
                (hit_ship.get_unit(), status)
            })
            .collect();
        hit_ships_status.iter().for_each(|(unit, status)| {
            unit.damage(
                status.damage,
                &status.engine_damage,
                &status.subsystem_damage,
            )
        });
        root.remove_dead();
        hit_ships_status
    }
}

impl ResourceProcess for StrategicWeapon {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
}

#[derive(Debug, Clone)]
pub struct FactoryClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    pub fn instantiate(class: Arc<Self>) -> Factory {
        Factory {
            class: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
            outputs: class.outputs.clone(),
        }
    }
}

impl PartialEq for FactoryClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FactoryClass {}

impl Ord for FactoryClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for FactoryClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for FactoryClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Factory {
    //this is an actual factory, derived from a factory class
    pub class: Arc<FactoryClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl Factory {
    //we take an active factory and update all its inputs and outputs to add or remove resources
    fn process(&mut self, location_efficiency: f32) {
        if let FactoryState::Active = self.get_state() {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.outputs
                .iter_mut()
                .for_each(|stockpile| stockpile.output_process(location_efficiency));
        } else {
        }
    }
}

impl ResourceProcess for Factory {
    //FactoryState determination method
    //this determines a factory's FactoryState based on the OutputStates of its outputs
    fn get_state(&self) -> FactoryState {
        let output_state: OutputState = if self.outputs.is_empty() {
            OutputState::Active //this ensures that a factory with no outputs will be considered active by default
        } else {
            self.outputs.iter().fold(OutputState::Dormant, |x, y| {
                //this makes a factory go dormant if all of its output stockpiles are dormant
                OutputState::reduce(x, y.output_state()) //this uses the OutputState reduce method defined earlier
            })
        };
        match output_state {
            OutputState::Active => {
                let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
                if input_is_good {
                    FactoryState::Active
                } else {
                    FactoryState::Stalled //even if the factory's outputs are fine, we stall the factory if it doesn't have enough resources in its input stockpiles
                }
            }
            OutputState::Dormant => FactoryState::Dormant, //here we just take the collapsed OutputState and make it the FactoryState
            OutputState::Stalled => FactoryState::Stalled,
        }
    }
    fn get_resource_supply_total(&self, resource: Arc<Resource>) -> u64 {
        let sum = self
            .outputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>();
        sum
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum::<u64>()
            + self.outputs.iter().map(|sp| sp.target).sum::<u64>()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FactoryState {
    Active,
    Dormant,
    Stalled,
}

pub enum OutputState {
    Active,
    Dormant,
    Stalled,
}

impl OutputState {
    //OutputState reduce method
    //this compares two stockpiles' OutputStates and collapses them in the appropriate manner; it's referenced later on in the FactoryState fold method
    fn reduce(a: Self, b: Self) -> Self {
        match a {
            OutputState::Active => match b {
                OutputState::Stalled => OutputState::Stalled,
                _ => OutputState::Active,
            },
            OutputState::Dormant => b,
            OutputState::Stalled => OutputState::Stalled,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShipyardClass {
    pub id: usize,
    pub visible_name: Option<String>,
    pub description: Option<String>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<ShipClassID, u64>,
    pub construct_rate: u64,
    pub efficiency: f32,
}

impl ShipyardClass {
    pub fn instantiate(class: Arc<Self>, shipclasses: &Vec<Arc<ShipClass>>) -> Shipyard {
        Shipyard {
            class: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
            outputs: class
                .outputs
                .iter()
                .map(|(shipclassid, num)| {
                    (
                        shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                        *num,
                    )
                })
                .collect(),
            construct_points: 0,
            efficiency: 1.0,
        }
    }
}

impl PartialEq for ShipyardClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipyardClass {}

impl Ord for ShipyardClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipyardClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipyardClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shipyard {
    pub class: Arc<ShipyardClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<Arc<ShipClass>, u64>,
    pub construct_points: u64,
    pub efficiency: f32,
}

impl Shipyard {
    fn process(&mut self, location_efficiency: f32) {
        if let FactoryState::Active = self.get_state() {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.construct_points +=
                (self.class.construct_rate as f32 * location_efficiency) as u64;
        }
    }

    fn try_choose_ship(&mut self, _shipclasses: &Vec<Arc<ShipClass>>) -> Option<Arc<ShipClass>> {
        //we go through the list of ships the shipyard can produce, specified as its outputs, and grab the one with the highest desirability weight
        let shipclass = self
            .outputs
            .iter()
            .max_by_key(|(_, weight)| *weight)
            .unwrap()
            .0;
        let cost = shipclass.base_strength;
        //then, if the shipyard has enough points to build it, we subtract the cost and return the ship class id
        if self.construct_points >= cost {
            self.construct_points -= cost;
            Some(shipclass.clone())
        //otherwise we return nothing
        } else {
            None
        }
    }

    //this uses try_choose_ship to generate the list of ships the shipyard is building this turn
    fn plan_ships(
        &mut self,
        location_efficiency: f32,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<Arc<ShipClass>> {
        self.process(location_efficiency);
        (0..)
            .map_while(|_| self.try_choose_ship(shipclasses))
            .collect()
    }
}

impl ResourceProcess for Shipyard {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
}

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
    fn get_unit_container_write(&self) -> RwLockWriteGuard<UnitContainer> {
        match self {
            UnitLocation::Node(node) => node.unit_container.write().unwrap(),
            UnitLocation::Squadron(squadron) => squadron.unit_container.write().unwrap(),
            UnitLocation::Hangar(hangar) => hangar.unit_container.write().unwrap(),
        }
    }
    fn get_ship_mut_read(&self) -> Option<RwLockReadGuard<ShipMut>> {
        match self {
            UnitLocation::Node(_) => None,
            UnitLocation::Squadron(_) => None,
            UnitLocation::Hangar(hangar) => Some(hangar.mother.mutables.read().unwrap()),
        }
    }
    fn is_alive_locked(
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
    fn check_insert(&self, container: &RwLockWriteGuard<UnitContainer>, unit: Unit) -> bool {
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
    fn check_remove(&self, container: &RwLockWriteGuard<UnitContainer>, unit: Unit) -> bool {
        container.contents.contains(&unit)
    }
    fn insert_unit(&self, container: &mut RwLockWriteGuard<UnitContainer>, unit: Unit) {
        container.contents.push(unit.clone());
    }
    fn remove_unit(&self, container: &mut RwLockWriteGuard<UnitContainer>, unit: Unit) {
        container.contents.retain(|content| content != &unit);
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
    fn get_morale_scalar(&self) -> f32;
    fn get_character_strength_scalar(&self) -> f32;
    fn get_interdiction_scalar(&self) -> f32;
    fn get_processordemandnavscalar(&self) -> f32;
    fn get_strength(&self, time: u64) -> u64;
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
                    ships_mut_lock.get_mut(&ship.id).unwrap().location = destination.clone()
                }
                None => {
                    squadrons_mut_lock.get_mut(&self.get_id()).unwrap().location =
                        destination.clone()
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
                        * self.get_processordemandnavscalar())
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
                            let engagement = root.internal_battle(EngagementPrep::engagement_prep(
                                root,
                                destination.clone(),
                                Some(aggressor),
                            ));
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
    fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.base_strength
            + self
                .hangars
                .iter()
                .map(|hangarclass| hangarclass.get_ideal_strength(root))
                .sum::<u64>()
    }
    fn get_ideal_volume(&self) -> u64 {
        self.hangar_vol
    }
    fn get_unitclass(class: Arc<Self>) -> UnitClass {
        UnitClass::ShipClass(class.clone())
    }
    //method to create a ship  with this ship class
    //to avoid an infinite loop, this does not give the ship any hangars
    //we generate those in a later step with build_hangars
    fn instantiate(
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
    fn new(val: u64) -> Self {
        ShipHealth { health: val }
    }
    fn get(&self) -> u64 {
        self.health
    }
    fn set(&mut self, val: u64) {
        if self.health > 0 {
            self.health = val
        }
    }
    fn add(&mut self, val: u64) {
        if self.health > 0 {
            self.health += val
        }
    }
    fn sub(&mut self, val: u64) {
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
    fn get_processordemandnavscalar(&self) -> f32 {
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
    pub disband_threshold: f32,
    pub deploys_self: bool, //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments
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
    fn get_processordemandnavscalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_processordemandnavscalar())
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
    fn new_from_unitclass(unitclass: &UnitClass) -> Self {
        match unitclass {
            UnitClass::ShipClass(sc) => UnitClassID::ShipClass(ShipClassID::new_from_index(sc.id)),
            UnitClass::SquadronClass(fc) => {
                UnitClassID::SquadronClass(SquadronClassID::new_from_index(fc.id))
            }
        }
    }
    fn get_index(&self) -> usize {
        match self {
            UnitClassID::ShipClass(sc) => sc.index,
            UnitClassID::SquadronClass(fc) => fc.index,
        }
    }
    fn get_unitclass(&self, root: &Root) -> UnitClass {
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
    fn get_processordemandnavscalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_processordemandnavscalar(),
            Unit::Squadron(squadron) => squadron.get_processordemandnavscalar(),
        }
    }
    fn get_strength(&self, time: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength(time),
            Unit::Squadron(squadron) => squadron.get_strength(time),
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct UnitRecord {
    pub id: u64,
    pub visible_name: String,
    pub class: UnitClass,
    pub allegiance: Arc<Faction>,
    pub daughters: Vec<u64>,
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

#[derive(Debug, PartialEq, Clone)]
pub struct FactionForces {
    pub local_forces: Vec<Unit>,
    pub reinforcements: Vec<(Arc<Node>, u64, Vec<Unit>)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FactionForcesRecord {
    pub local_forces: HashMap<UnitRecord, UnitStatus>,
    pub reinforcements: Vec<(Arc<Node>, u64, HashMap<UnitRecord, UnitStatus>)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct UnitStatus {
    pub location: Option<UnitLocation>,
    pub damage: i64,
    pub engine_damage: Vec<i64>,
    pub subsystem_damage: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct EngagementPrep {
    pub turn: u64,
    pub coalitions: HashMap<u64, HashMap<Arc<Faction>, FactionForces>>,
    pub wars: HashSet<(u64, u64)>,
    pub location: Arc<Node>,
    pub aggressor: Option<Arc<Faction>>,
}

impl EngagementPrep {
    pub fn engagement_prep(
        root: &Root,
        location: Arc<Node>,
        aggressor: Option<Arc<Faction>>,
    ) -> Self {
        let belligerents: HashMap<Arc<Faction>, Vec<Unit>> = location.clone().get_node_forces(root);

        let empty = &Vec::new();

        let neighbors: &Vec<Arc<Node>> = root.neighbors.get(&location).unwrap_or(empty);

        let mut coalition_counter = 0_u64;
        //a coalition is a set of factions which are not at war with each other and share all the same enemies
        let coalitions: HashMap<u64, HashMap<Arc<Faction>, FactionForces>> = belligerents
            .iter()
            .fold(HashMap::new(), |mut acc, (faction, _)| {
                if !acc
                    .iter()
                    .any(|(_, factions_map)| factions_map.iter().any(|(rhs, _)| rhs == faction))
                {
                    let enemy_factions: Vec<Arc<Faction>> = belligerents
                        .iter()
                        .filter(|(rhs, _)| {
                            root.wars.contains(&(
                                faction.clone().min(rhs.clone().clone()),
                                rhs.clone().max(&faction.clone().clone()).clone(),
                            ))
                        })
                        .map(|(faction, _)| faction.clone())
                        .collect();
                    let allied_factions: HashMap<Arc<Faction>, Vec<Unit>> = belligerents
                        .iter()
                        .filter(|(rhs, _)| !enemy_factions.contains(&rhs))
                        .filter(|(rhs, _)| {
                            enemy_factions.iter().all(|ghs| {
                                root.wars.contains(&(
                                    ghs.clone().min(rhs.clone().clone()),
                                    rhs.clone().max(&ghs.clone().clone()).clone(),
                                ))
                            })
                        })
                        .map(|(allied_faction, units)| {
                            (allied_faction.clone(), units.iter().cloned().collect())
                        })
                        .collect();
                    let allies_with_reinforcements: HashMap<Arc<Faction>, FactionForces> =
                        allied_factions
                            .iter()
                            .map(|(allied_faction, units)| {
                                let reinforcements = neighbors
                                    .iter()
                                    .map(|n| {
                                        (
                                            n.clone(),
                                            n.clone().get_distance(location.clone()),
                                            n.get_node_faction_reinforcements(
                                                location.clone(),
                                                faction.clone(),
                                                root,
                                            ),
                                        )
                                    })
                                    .collect();
                                (
                                    allied_faction.clone(),
                                    FactionForces {
                                        local_forces: units.iter().cloned().collect(),
                                        reinforcements: reinforcements,
                                    },
                                )
                            })
                            .collect();

                    acc.insert(coalition_counter, allies_with_reinforcements);
                    coalition_counter += 1;
                }
                acc
            });
        let wars = coalitions
            .iter()
            .map(|(index, faction_map)| {
                coalitions
                    .iter()
                    .fold(HashSet::new(), |mut acc, (rhs_index, rhs_faction_map)| {
                        let lhs = faction_map.keys().find(|_| true).unwrap();
                        let rhs = rhs_faction_map.keys().find(|_| true).unwrap();
                        if root
                            .wars
                            .contains(&(lhs.min(rhs).clone(), rhs.max(lhs).clone()))
                        {
                            acc.insert((index.min(rhs_index), rhs_index.max(index)));
                        }
                        acc
                    })
                    .iter()
                    .map(|(a, b)| (*a, *b))
                    .collect::<HashSet<_>>()
            })
            .flatten()
            .map(|(a, b)| (*a, *b))
            .collect();
        EngagementPrep {
            turn: root.turn.load(atomic::Ordering::Relaxed),
            coalitions,
            wars,
            location,
            aggressor,
        }
    }
    fn calculate_engagement_duration(&self, root: &Root, rng: &mut Hc128Rng) -> u64 {
        //we determine how long the battle lasts
        //taking into account both absolute and relative armada sizes
        //scaled logarithmically according to the specified exponent
        //as well as the scaling factors applied by the objectives of parties involved
        //then we multiply by a random number from a normal distribution
        let coalition_rough_strengths: HashMap<u64, i64> = self
            .coalitions
            .iter()
            .map(|(index, faction_map)| {
                (
                    *index,
                    faction_map
                        .iter()
                        .map(|(_, factionforces)| {
                            factionforces
                                .local_forces
                                .iter()
                                .map(|unit| {
                                    unit.get_strength(root.config.battle_scalars.avg_duration)
                                })
                                .sum::<u64>() //sum together all the unit strengths
                        })
                        .sum::<u64>() as i64, //then sum together the strengths for all the factions in the coalition
                )
            })
            .collect();

        let strongest_coalition: (&u64, &i64) = coalition_rough_strengths
            .iter()
            .max_by_key(|(_, strength)| *strength)
            .unwrap();

        let weaker_coalitions: HashMap<&u64, &i64> = coalition_rough_strengths
            .iter()
            .filter(|coalition| *coalition != strongest_coalition)
            .collect();

        let battle_size = coalition_rough_strengths
            .iter()
            .map(|(_, strength)| *strength)
            .sum::<i64>()
            - ((strongest_coalition.1
                - weaker_coalitions
                    .iter()
                    .map(|(_, strength)| *strength)
                    .sum::<i64>())
            .clamp(0, i64::MAX))
            .abs();

        let objective_duration_scalar: f32 = self
            .coalitions
            .iter()
            .map(|(_, faction_map)| {
                faction_map
                    .iter()
                    .map(|(_, factionforces)| {
                        factionforces
                            .local_forces
                            .iter()
                            .map(|unit| {
                                unit.get_objectives()
                                    .iter()
                                    .map(|objective| objective.duration_scalar)
                                    .product::<f32>() //we multiply the individual objectives' scalars
                            })
                            .product::<f32>() //then the units'
                    })
                    .product::<f32>() //then the factions'
            })
            .product::<f32>(); //and finally the coalitions'

        let duration: u64 =
            (((battle_size as f32).log(root.config.battle_scalars.duration_log_exp) + 300.0)
                * objective_duration_scalar
                * Normal::new(1.0, root.config.battle_scalars.duration_dev)
                    .unwrap()
                    .sample(rng))
            .clamp(0.0, 2.0) as u64;
        duration
    }
    fn get_coalition_strengths(&self, duration: u64) -> HashMap<u64, u64> {
        self.coalitions
            .iter()
            .map(|(index, faction_map)| {
                (
                    *index,
                    faction_map
                        .iter()
                        .map(|(_, forces)| {
                            let local_strength: u64 = forces
                                .local_forces
                                .iter()
                                .map(|unit| unit.get_strength(duration))
                                .sum();
                            let reinforcement_strength: u64 = forces
                                .reinforcements
                                .iter()
                                .map(|(_, lag, units)| {
                                    units
                                        .iter()
                                        .map(|unit| unit.get_strength(duration) as f32)
                                        .sum::<f32>()
                                        * ((duration.saturating_sub(*lag)) as f32 / duration as f32)
                                })
                                .sum::<f32>()
                                as u64;
                            local_strength + reinforcement_strength
                        })
                        .sum::<u64>(),
                )
            })
            .collect()
    }
    fn get_coalition_objective_difficulties(&self) -> HashMap<u64, f32> {
        self.coalitions
            .iter()
            .map(|(index, faction_map)| {
                (
                    *index,
                    faction_map
                        .iter()
                        .map(|(_, forces)| {
                            //we don't take the objectives of reinforcement units into account
                            forces
                                .local_forces
                                .iter()
                                .map(|unit| {
                                    unit.get_objectives()
                                        .iter()
                                        .map(|objective| objective.difficulty)
                                        .product::<f32>()
                                })
                                .product::<f32>()
                        })
                        .product(),
                )
            })
            .collect()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Engagement {
    pub visible_name: String,
    pub turn: u64,
    pub coalitions: HashMap<u64, HashMap<Arc<Faction>, FactionForces>>,
    pub aggressor: Option<Arc<Faction>>,
    pub objectives: HashMap<Arc<Faction>, Vec<Objective>>,
    pub location: Arc<Node>,
    pub duration: u64,
    pub victors: (Arc<Faction>, u64),
    pub unit_status: HashMap<u64, HashMap<Arc<Faction>, HashMap<Unit, UnitStatus>>>,
}

impl Engagement {
    pub fn battle_cleanup(&self, root: &Root) {
        println!("{}", self.visible_name);
        self.location.mutables.write().unwrap().allegiance = self.victors.0.clone();
        self.unit_status.iter().for_each(|(_, faction_map)| {
            faction_map.iter().for_each(|(_, unit_map)| {
                unit_map.iter().for_each(|(unit, status)| {
                    if let Some(place) = &status.location {
                        unit.damage(
                            status.damage,
                            &status.engine_damage,
                            &status.subsystem_damage,
                        );
                        unit.transfer(place.clone());
                        unit.repair(true);
                    } else {
                        unit.kill();
                    }
                })
            })
        });
        root.remove_dead();
        root.engagements.write().unwrap().push(self.record());
    }
    pub fn record(&self) -> EngagementRecord {
        EngagementRecord {
            visible_name: self.visible_name.clone(),
            turn: self.turn,
            coalitions: self
                .coalitions
                .iter()
                .map(|(id, faction_map)| {
                    (
                        *id,
                        faction_map
                            .iter()
                            .map(|(faction, forces)| {
                                (
                                    faction.clone(),
                                    FactionForcesRecord {
                                        local_forces: forces
                                            .local_forces
                                            .iter()
                                            .map(|unit| {
                                                (
                                                    unit.record(),
                                                    self.unit_status
                                                        .get(&id)
                                                        .unwrap()
                                                        .get(faction)
                                                        .unwrap()
                                                        .get(&unit)
                                                        .unwrap()
                                                        .clone(),
                                                )
                                            })
                                            .collect::<HashMap<_, _>>(),
                                        reinforcements: forces
                                            .reinforcements
                                            .iter()
                                            .map(|(node, distance, units)| {
                                                (
                                                    node.clone(),
                                                    *distance,
                                                    units
                                                        .iter()
                                                        .map(|unit| {
                                                            (
                                                                unit.record(),
                                                                self.unit_status
                                                                    .get(&id)
                                                                    .unwrap()
                                                                    .get(faction)
                                                                    .unwrap()
                                                                    .get(&unit)
                                                                    .unwrap()
                                                                    .clone(),
                                                            )
                                                        })
                                                        .collect(),
                                                )
                                            })
                                            .collect(),
                                    },
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            aggressor: self.aggressor.clone(),
            objectives: self.objectives.clone(),
            location: self.location.clone(),
            duration: self.duration.clone(),
            victors: self.victors.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct EngagementRecord {
    pub visible_name: String,
    pub turn: u64,
    pub coalitions: HashMap<u64, HashMap<Arc<Faction>, FactionForcesRecord>>,
    pub aggressor: Option<Arc<Faction>>,
    pub objectives: HashMap<Arc<Faction>, Vec<Objective>>,
    pub location: Arc<Node>,
    pub duration: u64,
    pub victors: (Arc<Faction>, u64),
}

//this takes an unbounded threat value and converts it to a multiplier between zero and one by which to scale saliences as they pass through the node
//uses scary math that Alyssa cooked up
//it's probably actually not that scary
fn scale_from_threat(threat: f32, scaling_factor: f32) -> f32 {
    if scaling_factor <= 0. {
        panic!(
            "Attempted to scale by nonpositive factor {}",
            scaling_factor
        );
    }

    let base = 0.95;
    let minimum_valid_threat_value = threat.abs() + scaling_factor;
    let downscaled_threat = minimum_valid_threat_value / scaling_factor;

    let delta_base = 1. - (1. / downscaled_threat);
    let maximum_delta = if threat >= 0. { 0.05 } else { -0.95 };
    let delta = maximum_delta * delta_base;

    base + delta
}

//polarity denotes whether a salience value represents supply or demand
//threat doesn't have demand
//at least not yet
//we might want to change that eventually
pub trait Polarity {}

//we put polarities in a dummy module for syntactic prettiness reasons
pub mod polarity {

    use super::Polarity;

    #[derive(Copy, Clone)]
    pub struct Supply {}

    impl Polarity for Supply {}

    #[derive(Copy, Clone)]
    pub struct Demand {}

    impl Polarity for Demand {}
}

pub trait Salience<P: Polarity> {
    //this retrieves the value of a specific salience in a specific node
    fn calculate_node_salience(
        self,
        root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        battle_duration: u64,
    ) -> Option<f32>;
}

//this method retrieves threat value generated by a given faction
impl Salience<polarity::Supply> for Arc<Faction> {
    fn calculate_node_salience(
        self,
        _root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        battle_duration: u64,
    ) -> Option<f32> {
        let node_strength: u64 = node.get_strength(self.clone(), battle_duration);
        //here we get the relations value -- the subject faction's opinion of the object faction, which will influence the threat value
        let relation = faction
            .relations
            .get(&FactionID::new_from_index(self.id))
            .unwrap();
        Some(node_strength)
            .filter(|&strength| strength != 0)
            .map(|strength| strength as f32 * relation * self.value_mult)
    }
}

impl Salience<polarity::Demand> for Arc<Faction> {
    fn calculate_node_salience(
        self,
        root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        //Here, we don't take the object faction's relations with the subject faction into account at all,
        //because subjectivity isn't really relevant for threat demand.
        //NOTE: Doing this by retrieving the data from salience state (rather than by looking at assets in the node directly) means that a node will want friendly forces corresponding to assets in other nodes
        //which are propagating salience into it. Not sure if this is desirable behavior.
        if faction == self {
            let resource_salience_by_node: Vec<Vec<[f32; 2]>> =
                transpose(&root.global_salience.resource_salience.read().unwrap()[self.id]);
            let resource_supply: f32 = resource_salience_by_node[node.id]
                .iter()
                .map(|array| array[0])
                .sum();
            let unitclass_salience_by_node: Vec<Vec<[f32; 2]>> =
                transpose(&root.global_salience.unitclass_salience.read().unwrap()[self.id]);
            let unitclass_supply: f32 = unitclass_salience_by_node[node.id]
                .iter()
                .map(|array| array[0])
                .sum();
            let node_value = resource_supply + unitclass_supply;
            let node_object_faction_supply =
                root.global_salience.faction_salience.read().unwrap()[self.id][self.id][node.id][0];
            Some(
                ((node_value
                    * self.volume_strength_ratio
                    * root.config.salience_scalars.volume_strength_ratio)
                    - node_object_faction_supply.clamp(0.0, f32::MAX))
                    * self.value_mult,
            )
        } else {
            None
        }
    }
}

//this method tells us how much supply there is of a given resource in a given node
impl Salience<polarity::Supply> for Arc<Resource> {
    fn calculate_node_salience(
        self,
        _root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        //NOTE: Currently this does not take input stockpiles of any kind into account. We may wish to change this.
        //we add up all the resource quantity in factory output stockpiles in the node
        let factorysupply: u64 = if node.mutables.read().unwrap().allegiance == faction {
            node.mutables
                .read()
                .unwrap()
                .factories
                .iter()
                .map(|factory| factory.get_resource_supply_total(self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //then all the valid resource quantity in units
        let shipsupply: u64 = node
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_resource_supply(self.clone()))
            .sum::<u64>();
        //then sum them together
        let sum = (factorysupply + shipsupply) as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.value_mult)
        }
    }
}

//this method tells us how much demand there is for a given resource in a given node
impl Salience<polarity::Demand> for Arc<Resource> {
    fn calculate_node_salience(
        self,
        _root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        //add up resources from factory input stockpiles in node
        let factorydemand: u64 = if node.mutables.read().unwrap().allegiance == faction {
            node.mutables
                .read()
                .unwrap()
                .factories
                .iter()
                .map(|factory| factory.get_resource_demand_total(self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //add up resources from shipyard input stockpiles in node
        let shipyarddemand: u64 = if node.mutables.read().unwrap().allegiance == faction {
            node.mutables
                .read()
                .unwrap()
                .shipyards
                .iter()
                .map(|shipyard| shipyard.get_resource_demand_total(self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //now we have to look at units in the node, since they might have stockpiles of their own
        let shipdemand: u64 = node
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_resource_demand(self.clone()))
            .sum::<u64>();
        //and sum everything together
        let sum = (factorydemand + shipyarddemand + shipdemand) as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.value_mult)
        }
    }
}

//this method tells us how much supply there is of a given shipclass in a given node
impl Salience<polarity::Supply> for UnitClass {
    fn calculate_node_salience(
        self,
        _root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        let sum = node
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_unitclass_supply_recursive(self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.get_value_mult())
        }
    }
}

//this method tells us how much demand there is for a given shipclass in a given node
impl Salience<polarity::Demand> for UnitClass {
    fn calculate_node_salience(
        self,
        _root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        let sum = node
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.is_alive())
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_unitclass_demand_recursive(self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.get_value_mult())
        }
    }
}

fn transpose<T>(v: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}

//TODO: make the logic apply more generally to stockpiles attached to ships

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalSalience {
    pub faction_salience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub resource_salience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub unitclass_salience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub strategic_weapon_effect_map: RwLock<Vec<Vec<[(i64, f32); 3]>>>,
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
            faction,
            self,
        ));
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
    pub fn internal_battle(&self, data: EngagementPrep) -> Engagement {
        let mut rng = Hc128Rng::seed_from_u64(47);

        let duration = data.calculate_engagement_duration(self, &mut rng);

        //we take the reinforcement data from our engagementprep and convert the distances to the scaling factor for travel time
        //for what percentage of the battle's duration the unit will be present
        let coalition_strengths = data.get_coalition_strengths(duration);

        let coalition_objective_difficulties = data.get_coalition_objective_difficulties();

        let coalition_chances: HashMap<u64, f32> = data
            .coalitions
            .iter()
            .map(|(index, faction_map)| {
                let chance: f32 = *coalition_strengths.get(index).unwrap() as f32
                    * *coalition_objective_difficulties.get(index).unwrap() as f32
                    * faction_map
                        .keys()
                        .map(|faction| faction.battle_scalar)
                        .product::<f32>()
                    * Normal::<f32>::new(1.0, self.config.battle_scalars.attacker_chance_dev)
                        .unwrap()
                        .sample(&mut rng)
                        .clamp(0.0, 2.0);
                (*index, chance)
            })
            .collect();

        let victor_coalition: u64 = *coalition_chances
            .iter()
            .max_by(|(_, chance), (_, rhs_chance)| chance.partial_cmp(rhs_chance).unwrap())
            .unwrap()
            .0;

        let neighbors = self.neighbors.get(&data.location).unwrap();

        let duration_damage_rand = Normal::<f32>::new(1.0, self.config.battle_scalars.damage_dev)
            .unwrap()
            .sample(&mut rng)
            .clamp(0.0, 1.0);

        //NOTE: Maybe have the lethality scaling over battle duration be logarithmic? Maybe modder-specified?
        let unit_status: HashMap<u64, HashMap<Arc<Faction>, HashMap<Unit, UnitStatus>>> = data
            .coalitions
            .iter()
            .map(|(index, faction_map)| {
                let is_victor = *index == victor_coalition;
                (
                    *index,
                    faction_map
                        .iter()
                        .map(|(faction, forces)| {
                            let all_faction_units: Vec<Unit> = forces
                                .local_forces
                                .iter()
                                .map(|unit| unit.get_daughters_recursive())
                                .chain(
                                    forces
                                        .reinforcements
                                        .iter()
                                        .map(|(_, _, units)| {
                                            units.iter().map(|unit| unit.get_daughters_recursive())
                                        })
                                        .flatten(),
                                )
                                .flatten()
                                .collect();
                            (
                                faction.clone(),
                                all_faction_units
                                    .iter()
                                    .map(|unit| {
                                        let new_location = match is_victor {
                                            true => {
                                                if unit.get_mother_node() == data.location {
                                                    unit.get_location()
                                                } else {
                                                    UnitLocation::Node(data.location.clone())
                                                }
                                            }
                                            false => {
                                                if unit.is_in_node() {
                                                    UnitLocation::Node(
                                                        unit.navigate(self, neighbors)
                                                            .unwrap_or(data.location.clone()),
                                                    )
                                                } else {
                                                    unit.get_location()
                                                }
                                            }
                                        };
                                        let allied_strength =
                                            *coalition_strengths.get(index).unwrap() as f32;
                                        let enemy_strength = coalition_strengths
                                            .iter()
                                            .filter(|(rhs_index, _)| {
                                                data.wars.contains(&(
                                                    *index.min(rhs_index),
                                                    **rhs_index.max(&index),
                                                ))
                                            })
                                            .map(|(_, strength)| *strength)
                                            .sum::<u64>()
                                            as f32;
                                        let (damage, engine_damage, strategic_weapon_damage) = unit
                                            .calculate_damage(
                                                self,
                                                is_victor,
                                                allied_strength,
                                                enemy_strength,
                                                duration,
                                                duration_damage_rand,
                                                &mut rng,
                                            );
                                        let is_alive = unit.get_hull() as i64 > damage;
                                        (
                                            unit.clone(),
                                            UnitStatus {
                                                location: match is_alive {
                                                    true => Some(new_location),
                                                    false => None,
                                                },
                                                damage,
                                                engine_damage,
                                                subsystem_damage: strategic_weapon_damage,
                                            },
                                        )
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                )
            })
            .collect();

        //NOTE: This isn't quite ideal -- we determine the victor faction by summing unit strengths, but we have to use this special-case method
        //that doesn't pay attention to daughters, because we're looking at all the units total --
        //so fleets are just counted as zero and we don't get any fleet modifiers
        let victor = unit_status
            .get(&victor_coalition)
            .unwrap()
            .iter()
            .max_by_key(|(_, unit_map)| {
                unit_map
                    .iter()
                    .filter(|(_, status)| status.location.is_some())
                    .map(|(unit, status)| unit.get_strength_post_engagement(status.damage))
                    .sum::<u64>()
            })
            .unwrap()
            .0
            .clone();

        Engagement {
            visible_name: format!("Battle of {}", data.location.visible_name.clone()),
            turn: data.turn,
            coalitions: data.coalitions,
            aggressor: data.aggressor.clone(),
            objectives: HashMap::new(),
            location: data.location,
            duration,
            victors: (victor, victor_coalition),
            unit_status,
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
    pub fn calculate_strategic_weapon_effect_map(&self) -> Vec<Vec<[(i64, f32); 3]>> {
        self.factions
            .iter()
            .map(|subject_faction| {
                self.ships
                    .read()
                    .unwrap()
                    .iter()
                    .filter(|ship| ship.mutables.read().unwrap().strategic_weapons.len() > 0)
                    .fold(
                        self.nodes.iter().map(|_| [(0, 0.0); 3]).collect(),
                        |mut acc: Vec<[(i64, f32); 3]>, ship| {
                            ship.mutables
                                .read()
                                .unwrap()
                                .strategic_weapons
                                .iter()
                                .filter(|weapon| {
                                    weapon.targets_faction(
                                        &self,
                                        &ship.get_allegiance(),
                                        subject_faction,
                                    )
                                })
                                .for_each(|weapon| {
                                    let damage = (
                                        (weapon.class.damage.0 .0 + weapon.class.damage.0 .1) / 2,
                                        (weapon.class.damage.1 .0 + weapon.class.damage.1 .1) / 2.0,
                                    );
                                    let engine_damage = (
                                        (weapon.class.engine_damage.0 .0
                                            + weapon.class.engine_damage.0 .1)
                                            / 2,
                                        (weapon.class.engine_damage.1 .0
                                            + weapon.class.engine_damage.1 .1)
                                            / 2.0,
                                    );
                                    let strategic_weapon_damage = (
                                        (weapon.class.strategic_weapon_damage.0 .0
                                            + weapon.class.strategic_weapon_damage.0 .1)
                                            / 2,
                                        (weapon.class.strategic_weapon_damage.1 .0
                                            + weapon.class.strategic_weapon_damage.1 .1)
                                            / 2.0,
                                    );
                                    let target_nodes = ship.get_mother_node().get_nodes_in_range(
                                        self,
                                        weapon.class.range,
                                        &weapon.class.forbidden_nodeflavors,
                                        &weapon.class.forbidden_edgeflavors,
                                    );
                                    target_nodes.iter().for_each(|node| {
                                        acc[node.id][0].0 += damage.0;
                                        acc[node.id][0].1 += damage.1;
                                        acc[node.id][1].0 += engine_damage.0;
                                        acc[node.id][1].1 += engine_damage.1;
                                        acc[node.id][2].0 += strategic_weapon_damage.0;
                                        acc[node.id][2].1 += strategic_weapon_damage.1;
                                    })
                                });
                            acc
                        },
                    )
            })
            .collect()
    }
    //oh god
    pub fn calculate_salience<S: Salience<P> + Clone, P: Polarity>(
        //we need a salience, which is the type of resource or shipclass or whatever we're calculating values for
        //and the faction for which we're calculating values
        //and we specify the number of times we want to calculate these values, (NOTE: uncertain) i.e. the number of edges we'll propagate across
        &self,
        salience: S,
        subject_faction: Arc<Faction>,
        deg_mult: f32,
        n_iters: usize,
    ) -> Vec<f32> {
        //this map only contains the salience values being generated by things directly in each node, without any propagation
        //we call get_value on the salience, and return the node id and salience value, while filtering down to only the nodes producing the subject salience
        //Length equals nodes producing subject salience
        let node_initial_salience_map: Vec<(Arc<Node>, f32)> = self
            .nodes
            .iter()
            .filter_map(|node| {
                salience
                    .clone()
                    .calculate_node_salience(
                        &self,
                        node.clone(),
                        subject_faction.clone(),
                        self.config.battle_scalars.avg_duration,
                    )
                    .map(|v| (node.clone(), v))
            })
            .collect();
        //this map contains the amount of threat that exists from each faction, in each node, from the perspective of the subject faction
        //Length equals all nodes
        //This is a subjective map for subject faction
        let tagged_threats: Vec<Vec<[f32; 2]>> =
            transpose(&self.global_salience.faction_salience.read().unwrap()[subject_faction.id]);
        //this is the factor by which a salience passing through each node should be multiplied
        //we sum the tagged threats for each node -- which are valenced according to relations with the subject faction
        //then we use Alyssa's black mathemagics to convert them so that the scaling curve is correct
        //Length equals all nodes
        //This is a subjective map for subject faction
        let node_degradations: Vec<f32> = tagged_threats
            .iter()
            .map(|factions_vec| {
                let sum = factions_vec.iter().map(|[supply, _]| supply).sum();
                scale_from_threat(sum, 20_f32) * deg_mult * 0.8
            })
            .collect();
        //Outer vec length equals all nodes; inner vec equals nodes owned by faction and producing specified salience -- but only the inner node corresponding to the outer node has a nonzero value
        let node_salience_state: Vec<Vec<f32>> = self
            .nodes
            .iter()
            .map(|node| {
                //we iterate over the node initial salience map, which contains only nodes owned by subject faction and producing subject salience
                node_initial_salience_map
                    .iter()
                    //that gives us the initial salience value for each node
                    //we use this '== check as u8' to multiply it by 1 if the node matches the one the outer iterator is looking at, and multiply it by 0 otherwise
                    .map(|(sourcenode, value)| {
                        value * ((sourcenode.clone() == node.clone()) as u8) as f32
                    })
                    .collect()
            })
            .collect();
        //this gives us a list of all nodes, with each node having an inner list of subject-subject nodes
        //if the node is a subject-subject node, then its inner list contains a self-entry indicating the salience it's generating, but all other values are zero
        //this will let us tag propagated saliences by their source so they don't improperly compound each other
        let n_tags = node_initial_salience_map.len();
        let node_salience_state =
            //there are three layers of iteration here
            //first we fold over node_salience_state as many times as specified by n_iters, defined earlier
            //then we iterate over all the edges
            //and inside that iteration we loop the salience propagation process as many times as there are relevant nodes (producing the salience and owned subject faction)
            //this way we can process each salience source separately, avoiding compounding
            (0..n_iters).fold(node_salience_state, |mut state, _| {
                //println!("Completed {} iterations of salience propagation.", n_iter);
                self.edges.iter().filter(|(_, flavor)| flavor.propagates).for_each(|((a, b), _)| {
                    //we get the degradation scalar for each of the two nodes in the edge
                    let deg_a = node_degradations[a.id];
                    let deg_b = node_degradations[b.id];
                    //this loop does basically the same thing as an iterator but we have to do it this way for complicated ownership reasons
                    //we repeat the loop process n_tags times, 
                    for i in 0..n_tags {
                        //we index into node_salience_state's outer vec by node A's id, then into the inner vec by i; this means we're essentially iterating over the inner vec
                        //we update the i'th element of A (the inner vec) by taking the maximum between the i'th element of A and the i'th element of B, multiplied by node B's degradation scalar
                        //because this is the salience coming from node B to node A, getting degraded by B's threats as it leaves
                        state[a.id][i] = state[a.id][i].max(state[b.id][i] * deg_b);
                        //then we do the same thing again but backwards, to process the salience coming from node A to node B
                        state[b.id][i] = state[b.id][i].max(state[a.id][i] * deg_a);
                    }
                });
                //now we return the new state, with it having been updated
                state
            });
        //now we collapse all the different producers together so we can just look at how much salience each node contains from the perspective of the subject faction
        node_salience_state
            .iter()
            .map(|salience| salience.iter().sum())
            .collect()
    }
    pub fn calculate_global_faction_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .par_iter()
            .map(|subjectfaction| {
                self.factions
                    .par_iter()
                    .map(|objectfaction| {
                        if objectfaction.propagates {
                            let supply = self.calculate_salience::<Arc<Faction>, polarity::Supply>(
                                objectfaction.clone(),
                                subjectfaction.clone(),
                                self.config.salience_scalars.faction_deg_mult,
                                self.config.salience_scalars.faction_prop_iters,
                            );
                            let demand = self.calculate_salience::<Arc<Faction>, polarity::Demand>(
                                objectfaction.clone(),
                                subjectfaction.clone(),
                                self.config.salience_scalars.faction_deg_mult,
                                self.config.salience_scalars.faction_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    })
                    .collect()
            })
            .collect()
    }
    pub fn calculate_global_resource_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .par_iter()
            .map(|faction| {
                self.resources
                    .par_iter()
                    .map(|resource| {
                        if resource.propagates {
                            let supply = self
                                .calculate_salience::<Arc<Resource>, polarity::Supply>(
                                    resource.clone(),
                                    faction.clone(),
                                    self.config.salience_scalars.resource_deg_mult,
                                    self.config.salience_scalars.resource_prop_iters,
                                );
                            let demand = self
                                .calculate_salience::<Arc<Resource>, polarity::Demand>(
                                    resource.clone(),
                                    faction.clone(),
                                    self.config.salience_scalars.resource_deg_mult,
                                    self.config.salience_scalars.resource_prop_iters,
                                );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    })
                    .collect()
            })
            .collect()
    }
    pub fn calculate_global_unitclass_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .par_iter()
            .map(|faction| {
                self.shipclasses
                    .par_iter()
                    .map(|shipclass| {
                        if shipclass.propagates {
                            let supply = self.calculate_salience::<UnitClass, polarity::Supply>(
                                ShipClass::get_unitclass(shipclass.clone()),
                                faction.clone(),
                                self.config.salience_scalars.unitclass_deg_mult,
                                self.config.salience_scalars.unitclass_prop_iters,
                            );
                            let demand = self.calculate_salience::<UnitClass, polarity::Demand>(
                                ShipClass::get_unitclass(shipclass.clone()),
                                faction.clone(),
                                self.config.salience_scalars.unitclass_deg_mult,
                                self.config.salience_scalars.unitclass_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    })
                    .chain(self.squadronclasses.par_iter().map(|squadronclass| {
                        if squadronclass.propagates {
                            let supply = self.calculate_salience::<UnitClass, polarity::Supply>(
                                SquadronClass::get_unitclass(squadronclass.clone()),
                                faction.clone(),
                                self.config.salience_scalars.unitclass_deg_mult,
                                self.config.salience_scalars.unitclass_prop_iters,
                            );
                            let demand = self.calculate_salience::<UnitClass, polarity::Demand>(
                                SquadronClass::get_unitclass(squadronclass.clone()),
                                faction.clone(),
                                self.config.salience_scalars.unitclass_deg_mult,
                                self.config.salience_scalars.unitclass_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    }))
                    .collect()
            })
            .collect()
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
                self.create_squadron(id.clone(), location.clone(), faction.clone())
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
    use super::scale_from_threat;
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
