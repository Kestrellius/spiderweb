use crate::internal::faction::Faction;
use crate::internal::node::{Locality, Node};
use crate::internal::root::{Objective, Root};
use crate::internal::unit::{Mobility, Unit, UnitClass, UnitLocation};
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

impl UnitRecord {
    pub fn get_unit(&self, root: &Root) -> Option<Unit> {
        //find is slow, which may or may not matter here
        //consider switching the ships and squadrons vecs to a hashmap of u64 to unit?
        match self.class {
            UnitClass::ShipClass(_) => root
                .ships
                .read()
                .unwrap()
                .iter()
                .find(|ship| ship.id == self.id)
                .map(|ship| ship.get_unit()),
            UnitClass::SquadronClass(_) => root
                .squadrons
                .read()
                .unwrap()
                .iter()
                .find(|squadron| squadron.id == self.id)
                .map(|squadron| squadron.get_unit()),
        }
    }
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
    pub fn internal_battle(&self, root: &Root) -> Engagement {
        let mut rng = Hc128Rng::seed_from_u64(47);

        let duration = self.calculate_engagement_duration(root, &mut rng);

        //we take the reinforcement data from our engagementprep and convert the distances to the scaling factor for travel time
        //for what percentage of the battle's duration the unit will be present
        let coalition_strengths = self.get_coalition_strengths(duration);

        let coalition_objective_difficulties = self.get_coalition_objective_difficulties();

        let coalition_chances: HashMap<u64, f32> = self
            .coalitions
            .iter()
            .map(|(index, faction_map)| {
                let chance: f32 = *coalition_strengths.get(index).unwrap() as f32
                    * *coalition_objective_difficulties.get(index).unwrap() as f32
                    * faction_map
                        .keys()
                        .map(|faction| faction.battle_scalar)
                        .product::<f32>()
                    * Normal::<f32>::new(1.0, root.config.battle_scalars.attacker_chance_dev)
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

        let neighbors = root.neighbors.get(&self.location).unwrap();

        let duration_damage_rand = Normal::<f32>::new(1.0, root.config.battle_scalars.damage_dev)
            .unwrap()
            .sample(&mut rng)
            .clamp(0.0, 1.0);

        //NOTE: Maybe have the lethality scaling over battle duration be logarithmic? Maybe modder-specified?
        let unit_status: HashMap<u64, HashMap<Arc<Faction>, HashMap<Unit, UnitStatus>>> = self
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
                                                if unit.get_mother_node() == self.location {
                                                    unit.get_location()
                                                } else {
                                                    UnitLocation::Node(self.location.clone())
                                                }
                                            }
                                            false => {
                                                if unit.is_in_node() {
                                                    UnitLocation::Node(
                                                        unit.navigate(root, neighbors, true)
                                                            .expect("Defeated ship is attempting to stay on battlefield! Investigate."),
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
                                                self.wars.contains(&(
                                                    *index.min(rhs_index),
                                                    **rhs_index.max(&index),
                                                ))
                                            })
                                            .map(|(_, strength)| *strength)
                                            .sum::<u64>()
                                            as f32;
                                        let (damage, engine_damage, strategic_weapon_damage) = unit
                                            .calculate_damage(
                                                root,
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
            visible_name: format!("Battle of {}", self.location.visible_name.clone()),
            turn: self.turn,
            coalitions: self.coalitions.clone(),
            aggressor: self.aggressor.clone(),
            objectives: HashMap::new(),
            location: self.location.clone(),
            duration,
            victors: (victor, victor_coalition),
            unit_status,
        }
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
