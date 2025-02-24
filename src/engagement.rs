use crate::faction::Faction;
use crate::node::{Locality, Node};
use crate::root::{Objective, Root};
use crate::unit::{Mobility, Unit, UnitClass, UnitLocation};
use rand::prelude::*;
use rand_distr::*;
use rand_hc::Hc128Rng;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::atomic::{self};
use std::sync::Arc;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct UnitRecord {
    pub id: u64,
    pub visible_name: String,
    pub class: UnitClass,
    pub allegiance: Arc<Faction>,
    pub daughters: Vec<u64>,
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
    pub fn calculate_engagement_duration(&self, root: &Root, rng: &mut Hc128Rng) -> u64 {
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
    pub fn get_coalition_strengths(&self, duration: u64) -> HashMap<u64, u64> {
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
    pub fn get_coalition_objective_difficulties(&self) -> HashMap<u64, f32> {
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
