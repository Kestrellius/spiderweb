use crate::internal::faction::Faction;
use crate::internal::hangar::{Hangar, UnitContainer};
use crate::internal::resource::{
    Factory, Resource, ResourceProcess, Shipyard, Stockpileness, UnipotentStockpile,
};
use crate::internal::root::Root;
use crate::internal::unit::{
    Mobility, Ship, ShipClass, ShipMut, Squadron, SquadronClass, SquadronMut, Unit, UnitClassID,
    UnitLocation,
};
use itertools::Itertools;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{self};
use std::sync::Arc;
use std::sync::{RwLock, RwLockWriteGuard};

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
    pub fn get_resource_supply(&self, faction: Arc<Faction>, resource: Arc<Resource>) -> u64 {
        //NOTE: Currently this does not take input stockpiles of any kind into account. We may wish to change this.
        //we add up all the resource quantity in factory output stockpiles in the node
        let factorysupply: u64 = if self.mutables.read().unwrap().allegiance == faction {
            self.mutables
                .read()
                .unwrap()
                .factories
                .iter()
                .map(|factory| factory.get_resource_supply_total(resource.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //then all the valid resource quantity in units
        let shipsupply: u64 = self
            .unit_container
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_resource_supply(resource.clone()))
            .sum::<u64>();
        //then sum them together
        factorysupply + shipsupply
    }
    pub fn get_strength(&self, faction: Arc<Faction>, time: u64) -> u64 {
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
    pub fn get_cluster(&self, root: &Root) -> Option<Arc<Cluster>> {
        let cluster = root.clusters.iter().find(|cluster| {
            cluster
                .nodes
                .iter()
                .find(|sys_node| sys_node.id == self.id)
                .is_some()
        });
        match cluster {
            Some(sys) => Some(sys.clone()),
            None => None,
        }
    }
    pub fn is_in_cluster(&self, cluster: Arc<Cluster>) -> bool {
        cluster
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

pub trait Locality {
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
    }

    //absolute horrorshow, never think about
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

            //we have to acquire write locks on every unit-related rwlock in this node,
            //because this needs to be thread-safe, and if we didn't have locks for the entire duration,
            //then between the time we figured out how many of what unit everything needed,
            //and the time we transferred things,
            //another thread might come along and change the configuration of things in the node.
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

            //we need to know the speed of the ships hangars are attached to,
            //because it's going to be relevant later for figuring out how much demand we allot
            //hangars that are purely for transporting ships
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

            struct UnitClassTransactionData {
                supply_demand_ratio: f32,
                squadrons_demand: HashMap<u64, u64>,
                hangars_demand: HashMap<u64, u64>,
            }

            let transaction_data_by_unitclass: Vec<UnitClassTransactionData> = unitclasses
                .iter()
                .map(|unitclass| {
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
                                + total_transport_naive_demand)
                                as f32)
                            .clamp(0.0, f32::MAX);

                    let transport_hangars_demand: HashMap<u64, u64> =
                        all_hangars_indexed_with_speed
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

                    let hangars_demand: HashMap<u64, u64> = non_transport_hangars_demand
                        .iter()
                        .chain(transport_hangars_demand.iter())
                        .map(|(a, b)| (*a, *b))
                        .collect();

                    let supply_total: f32 = (node_supply
                        + squadrons_supply
                            .iter()
                            .map(|(_, supply)| *supply)
                            .sum::<u64>()
                        + hangars_supply
                            .iter()
                            .map(|(_, supply)| *supply)
                            .sum::<u64>()) as f32;

                    let demand_total: f32 = (squadrons_demand
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
                            .sum::<u64>()) as f32;

                    let num = all_units
                        .iter()
                        .filter(|unit| &unit.get_unitclass() == unitclass)
                        .count() as u64;

                    UnitClassTransactionData {
                        supply_demand_ratio: (supply_total / demand_total),
                        squadrons_demand,
                        hangars_demand,
                    }
                })
                .collect();

            struct ProperVolumePrep {
                proper_ideal_cargo_volume: u64,
                demand_left: u64,
            }

            let squadrons_proper_volume_preps: HashMap<u64, Vec<ProperVolumePrep>> =
                all_squadrons_indexed
                    .iter()
                    .map(|(index, squadron)| {
                        let proper_ideal_cargo_volumes_by_unitclass = unitclasses
                            .iter()
                            .enumerate()
                            .map(|(unitclass_index, unitclass)| {
                                let transaction_data =
                                    &transaction_data_by_unitclass[unitclass_index];
                                let allowed_cargo_volume: u64 =
                                    (*transaction_data.squadrons_demand.get(index).unwrap() as f32
                                        * transaction_data.supply_demand_ratio)
                                        as u64;

                                let squadron_demand =
                                    *transaction_data.squadrons_demand.get(index).unwrap() as f32;
                                let proper_ideal_cargo_volume = (squadron
                                    .class
                                    .ideal
                                    .get(&UnitClassID::new_from_unitclass(&unitclass))
                                    .unwrap_or(&0)
                                    * unitclass.get_ideal_volume())
                                .min(allowed_cargo_volume);
                                let class_units = all_units
                                    .iter()
                                    .filter(|unit| &unit.get_unitclass() == unitclass)
                                    .collect::<Vec<_>>();
                                let mother_loyalty_average = class_units
                                    .iter()
                                    .map(|unit| {
                                        let is_loyal = match unit {
                                            Unit::Ship(_) => {
                                                all_ships_mut_lock
                                                    .get(&unit.get_id())
                                                    .unwrap()
                                                    .last_mother
                                                    == Some(squadron.id)
                                            }
                                            Unit::Squadron(_) => {
                                                all_squadrons_mut_lock
                                                    .get(&unit.get_id())
                                                    .unwrap()
                                                    .last_mother
                                                    == Some(squadron.id)
                                            }
                                        };
                                        (unit.get_mother_loyalty_scalar() * (is_loyal as i8 as f32))
                                            + (1.0 * (!is_loyal as i8 as f32))
                                    })
                                    .sum::<f32>()
                                    / class_units.len() as f32;
                                ProperVolumePrep {
                                    proper_ideal_cargo_volume,
                                    demand_left: ((squadron_demand as u64)
                                        .saturating_sub(proper_ideal_cargo_volume)
                                        as f32
                                        * mother_loyalty_average)
                                        as u64,
                                }
                            })
                            .collect::<Vec<_>>();
                        (*index, proper_ideal_cargo_volumes_by_unitclass)
                    })
                    .collect();

            let hangars_proper_volume_preps: HashMap<u64, Vec<ProperVolumePrep>> =
                all_hangars_indexed_with_speed
                    .iter()
                    .map(|(index, (hangar, _))| {
                        let proper_ideal_cargo_volumes_by_unitclass = unitclasses
                            .iter()
                            .enumerate()
                            .map(|(unitclass_index, unitclass)| {
                                let transaction_data =
                                    &transaction_data_by_unitclass[unitclass_index];
                                let allowed_cargo_volume: u64 =
                                    (*transaction_data.hangars_demand.get(index).unwrap() as f32
                                        * transaction_data.supply_demand_ratio)
                                        as u64;
                                let hangar_demand =
                                    *transaction_data.hangars_demand.get(index).unwrap() as f32;
                                let proper_ideal_cargo_volume = (hangar
                                    .class
                                    .ideal
                                    .get(&UnitClassID::new_from_unitclass(&unitclass))
                                    .unwrap_or(&0)
                                    * unitclass.get_ideal_volume())
                                .min(allowed_cargo_volume);
                                let class_units = all_units
                                    .iter()
                                    .filter(|unit| &unit.get_unitclass() == unitclass)
                                    .collect::<Vec<_>>();
                                let mother_loyalty_average = class_units
                                    .iter()
                                    .map(|unit| {
                                        let is_loyal = match unit {
                                            Unit::Ship(_) => {
                                                all_ships_mut_lock
                                                    .get(&unit.get_id())
                                                    .unwrap()
                                                    .last_mother
                                                    == Some(hangar.mother.id)
                                            }
                                            Unit::Squadron(_) => {
                                                all_squadrons_mut_lock
                                                    .get(&unit.get_id())
                                                    .unwrap()
                                                    .last_mother
                                                    == Some(hangar.mother.id)
                                            }
                                        };
                                        (unit.get_mother_loyalty_scalar() * (is_loyal as i8 as f32))
                                            + (1.0 * (!is_loyal as i8 as f32))
                                    })
                                    .sum::<f32>()
                                    / class_units.len() as f32;
                                ProperVolumePrep {
                                    proper_ideal_cargo_volume,
                                    demand_left: ((hangar_demand as u64)
                                        .saturating_sub(proper_ideal_cargo_volume)
                                        as f32
                                        * mother_loyalty_average)
                                        as u64,
                                }
                            })
                            .collect::<Vec<_>>();
                        (*index, proper_ideal_cargo_volumes_by_unitclass)
                    })
                    .collect();

            let squadrons_proper_cargo_volumes: HashMap<u64, Vec<u64>> = all_squadrons_indexed
                .iter()
                .map(|(index, squadron)| {
                    let proper_cargo_volumes_by_unitclass = unitclasses
                        .iter()
                        .enumerate()
                        .map(|(unitclass_index, unitclass)| {
                            let transaction_data = &transaction_data_by_unitclass[unitclass_index];
                            let space_left = squadron.class.capacity.saturating_sub(
                                squadrons_proper_volume_preps
                                    .get(index)
                                    .unwrap()
                                    .iter()
                                    .map(|prep| prep.proper_ideal_cargo_volume)
                                    .sum(),
                            );
                            let total_demand_left: u64 = squadrons_proper_volume_preps
                                .get(index)
                                .unwrap()
                                .iter()
                                .map(|prep| prep.demand_left)
                                .sum();
                            let space_allowance_ratio =
                                space_left as f32 / total_demand_left as f32;
                            let proper_volume_prep =
                                &squadrons_proper_volume_preps.get(index).unwrap()[unitclass_index];
                            let desired_cargo_volume = proper_volume_prep.proper_ideal_cargo_volume
                                + (proper_volume_prep.demand_left as f32 * space_allowance_ratio)
                                    as u64;
                            let allowed_cargo_volume: u64 =
                                (*transaction_data.squadrons_demand.get(index).unwrap() as f32
                                    * transaction_data.supply_demand_ratio)
                                    as u64;

                            desired_cargo_volume.min(allowed_cargo_volume)
                        })
                        .collect::<Vec<_>>();
                    (*index, proper_cargo_volumes_by_unitclass)
                })
                .collect();

            let hangars_proper_cargo_volumes: HashMap<u64, Vec<u64>> =
                all_hangars_indexed_with_speed
                    .iter()
                    .map(|(index, (hangar, _))| {
                        let proper_cargo_volumes_by_unitclass = unitclasses
                            .iter()
                            .enumerate()
                            .map(|(unitclass_index, _)| {
                                let transaction_data =
                                    &transaction_data_by_unitclass[unitclass_index];
                                let space_left = hangar.class.capacity.saturating_sub(
                                    hangars_proper_volume_preps
                                        .get(index)
                                        .unwrap()
                                        .iter()
                                        .map(|prep| prep.proper_ideal_cargo_volume)
                                        .sum(),
                                );
                                let total_demand_left: u64 = hangars_proper_volume_preps
                                    .get(index)
                                    .unwrap()
                                    .iter()
                                    .map(|prep| prep.demand_left)
                                    .sum();
                                let space_allowance_ratio =
                                    space_left as f32 / total_demand_left as f32;
                                let proper_volume_prep = &hangars_proper_volume_preps
                                    .get(index)
                                    .unwrap()[unitclass_index];
                                let desired_cargo_volume = proper_volume_prep
                                    .proper_ideal_cargo_volume
                                    + (proper_volume_prep.demand_left as f32
                                        * space_allowance_ratio)
                                        as u64;
                                let allowed_cargo_volume: u64 =
                                    (*transaction_data.hangars_demand.get(index).unwrap() as f32
                                        * transaction_data.supply_demand_ratio)
                                        as u64;
                                desired_cargo_volume.min(allowed_cargo_volume)
                            })
                            .collect::<Vec<_>>();
                        (*index, proper_cargo_volumes_by_unitclass)
                    })
                    .collect();

            unitclasses.iter().for_each(|unitclass| {
                all_squadrons_indexed.iter().for_each(|(index, squadron)| {
                    let container = all_squadrons_containers_lock.get(index).unwrap();
                    let proper_cargo_volume: u64 = squadrons_proper_cargo_volumes
                        .get(&squadron.id)
                        .unwrap()[unitclass.get_id()];
                    let mut volume_transferred: u64 = 0;
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
                    let squadron_initial_cargo_volume: u64 = unit_volumes.values().sum();
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
                        if squadron_initial_cargo_volume
                            .saturating_sub(volume_transferred)
                            .saturating_sub(unit_volume / 2)
                            < proper_cargo_volume
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
                        volume_transferred += unit_volume;
                    }
                });

                all_hangars_indexed_with_speed
                    .iter()
                    .for_each(|(index, (hangar, _))| {
                        let container = all_hangars_containers_lock.get(index).unwrap();
                        let proper_cargo_volume: u64 = hangars_proper_cargo_volumes
                            .get(&hangar.id)
                            .unwrap()[unitclass.get_id()];
                        let mut volume_transferred: u64 = 0;
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
                        let hangar_initial_cargo_volume: u64 = unit_volumes.values().sum();
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
                            if hangar_initial_cargo_volume
                                .saturating_sub(volume_transferred)
                                .saturating_sub(unit_volume / 2)
                                < proper_cargo_volume
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
                            volume_transferred += unit_volume;
                        }
                    });

                all_squadrons_indexed.iter().for_each(|(index, squadron)| {
                    let squadron_container = all_squadrons_containers_lock.get(index).unwrap();
                    let squadron_initial_cargo_volume: u64 = squadron_container
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
                    let proper_cargo_volume: u64 = squadrons_proper_cargo_volumes
                        .get(&squadron.id)
                        .unwrap()[unitclass.get_id()];
                    let mut volume_transferred: u64 = 0;
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
                        .sorted_by_key(|unit| {
                            NotNan::new(
                                -(unit.get_mother_loyalty_scalar()
                                    * (match unit {
                                        Unit::Ship(_) => {
                                            all_ships_mut_lock
                                                .get(&unit.get_id())
                                                .unwrap()
                                                .last_mother
                                                == Some(squadron.id)
                                        }
                                        Unit::Squadron(_) => {
                                            all_squadrons_mut_lock
                                                .get(&unit.get_id())
                                                .unwrap()
                                                .last_mother
                                                == Some(squadron.id)
                                        }
                                    } as i8 as f32)),
                            )
                            .unwrap()
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    for contained_unit in relevant_node_container_contents {
                        let unit_volume = unit_volumes.get(&contained_unit.get_id()).unwrap();
                        if squadron_initial_cargo_volume + (volume_transferred) + (unit_volume / 2)
                            > proper_cargo_volume
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
                        volume_transferred += unit_volume;
                    }
                });

                all_hangars_indexed_with_speed
                    .iter()
                    .for_each(|(index, (hangar, _))| {
                        let hangar_container = all_hangars_containers_lock.get(index).unwrap();
                        let hangar_initial_cargo_volume: u64 = hangar_container
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
                        let proper_cargo_volume: u64 = hangars_proper_cargo_volumes
                            .get(&hangar.id)
                            .unwrap()[unitclass.get_id()];
                        let mut volume_transferred: u64 = 0;
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
                            .sorted_by_key(|unit| {
                                NotNan::new(
                                    -(unit.get_mother_loyalty_scalar()
                                        * (match unit {
                                            Unit::Ship(_) => {
                                                all_ships_mut_lock
                                                    .get(&unit.get_id())
                                                    .unwrap()
                                                    .last_mother
                                                    == Some(hangar.mother.id)
                                            }
                                            Unit::Squadron(_) => {
                                                all_squadrons_mut_lock
                                                    .get(&unit.get_id())
                                                    .unwrap()
                                                    .last_mother
                                                    == Some(hangar.mother.id)
                                            }
                                        } as i8 as f32)),
                                )
                                .unwrap()
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        for contained_unit in relevant_node_container_contents {
                            let unit_volume = unit_volumes.get(&contained_unit.get_id()).unwrap();
                            if hangar_initial_cargo_volume
                                + (volume_transferred)
                                + (unit_volume / 2)
                                > proper_cargo_volume
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
                            volume_transferred += unit_volume;
                        }
                    });
            });
            all_squadrons_indexed.iter().for_each(|(index, squadron)| {
                if squadron.get_strength_locked(
                    &all_hangars_containers_lock,
                    &all_squadrons_containers_lock,
                    &all_ships_mut_lock,
                    &all_squadrons_mut_lock,
                    &squadron_container_is_not_empty_map,
                    root.config.battle_scalars.avg_duration,
                ) as f32
                    > (squadron.class.get_ideal_strength(root) as f32
                        * squadron.class.de_ghost_threshold)
                {
                    all_squadrons_mut_lock.get_mut(index).unwrap().ghost = false
                }
            });
        });
        node_mut.units_transacted = true;
    }
}

#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: usize,
    pub nodes: Vec<Arc<Node>>,
}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Cluster {}

impl Ord for Cluster {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Cluster {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Cluster {
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
    neighbormap: HashMap<Arc<Node>, Vec<Arc<Node>>>, //NOTE: investigate. Map of which nodes belong to which clusters, for purposes of generating all-to-all links
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
