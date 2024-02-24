//this is the section of the program that manages the json files defined by the modder
use crate::internal;
use itertools::Itertools;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::iter;
use std::sync::atomic::{self, AtomicU64, AtomicUsize};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    salience_scalars: SalienceScalars,
    entity_scalars: EntityScalars,
    battle_scalars: BattleScalars,
}

impl Config {
    fn hydrate(self) -> internal::Config {
        internal::Config {
            salience_scalars: self.salience_scalars.hydrate(),
            entity_scalars: self.entity_scalars.hydrate(),
            battle_scalars: self.battle_scalars.hydrate(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SalienceScalars {
    faction_deg_mult: Option<f32>,
    resource_deg_mult: Option<f32>,
    unitclass_deg_mult: Option<f32>,
    faction_prop_iters: Option<usize>, //number of edges across which this salience will propagate during a turn
    resource_prop_iters: Option<usize>,
    unitclass_prop_iters: Option<usize>,
    volume_strength_ratio: Option<f32>,
}

impl SalienceScalars {
    fn hydrate(self) -> internal::SalienceScalars {
        internal::SalienceScalars {
            faction_deg_mult: self.faction_deg_mult.unwrap_or(0.5),
            resource_deg_mult: self.resource_deg_mult.unwrap_or(0.5),
            unitclass_deg_mult: self.unitclass_deg_mult.unwrap_or(0.5),
            faction_prop_iters: self.faction_prop_iters.unwrap_or(5),
            resource_prop_iters: self.resource_prop_iters.unwrap_or(5),
            unitclass_prop_iters: self.unitclass_prop_iters.unwrap_or(5),
            volume_strength_ratio: self.volume_strength_ratio.unwrap_or(0.5),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntityScalars {
    avg_speed: Option<u64>,
    defect_escape_scalar: Option<f32>,
    victor_morale_scalar: Option<f32>,
    victis_morale_scalar: Option<f32>,
}

impl EntityScalars {
    fn hydrate(self) -> internal::EntityScalars {
        internal::EntityScalars {
            avg_speed: self.avg_speed.unwrap_or(1),
            defect_escape_scalar: self.defect_escape_scalar.unwrap_or(1.0),
            victor_morale_scalar: self.victor_morale_scalar.unwrap_or(1.0),
            victis_morale_scalar: self.victis_morale_scalar.unwrap_or(1.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BattleScalars {
    avg_duration: Option<u64>, //average battle duration, used for strength calculations; defaults to 600
    duration_log_exp: Option<f32>, //logarithmic exponent for scaling of battle duration over battle size; defaults to 1.0025
    duration_dev: Option<f32>, //standard deviation for the randomly-generated scaling factor for battle duration; defaults to 0.25
    attacker_chance_dev: Option<f32>, //standard deviation for the randomly-generated scaling factor for the attackers' chance of winning a battle; defaults to 0.25
    defender_chance_dev: Option<f32>, //standard deviation for the randomly-generated scaling factor for the defenders' chance of winning a battle; defaults to 0.25
    vae_victor: Option<f32>, //multiplier for damage done to ships winning a battle; defaults to 0.5
    vae_victis: Option<f32>, //multiplier for damage done to ships losing a battle; defaults to 1.0
    damage_dev: Option<f32>, //standard deviation for the randomly-generated scaling factor for damage done to ships; defaults to 1.005
    base_damage: Option<f32>, //base value for the additive damage done to ships in addition to the percentage-based damage; defaults to 50.0
    engine_damage_scalar: Option<f32>, //multiplier for damage done to ships' engines
    strategic_weapon_damage_scalar: Option<f32>, //multiplier for damage done to strategic weapons on ships
    duration_damage_scalar: Option<f32>, //multiplier for damage increase as battle duration rises; defaults to 1.0
}

impl BattleScalars {
    fn hydrate(self) -> internal::BattleScalars {
        internal::BattleScalars {
            avg_duration: self.avg_duration.unwrap_or(600),
            duration_log_exp: self.duration_log_exp.unwrap_or(1.0025),
            duration_dev: self.duration_dev.unwrap_or(0.25),
            attacker_chance_dev: self.attacker_chance_dev.unwrap_or(0.25),
            defender_chance_dev: self.defender_chance_dev.unwrap_or(0.25),
            vae_victor: self.vae_victor.unwrap_or(0.5),
            vae_victis: self.vae_victis.unwrap_or(1.0),
            damage_dev: self.damage_dev.unwrap_or(1.005),
            base_damage: self.base_damage.unwrap_or(50.0),
            engine_damage_scalar: self.engine_damage_scalar.unwrap_or(0.1),
            strategic_weapon_damage_scalar: self.strategic_weapon_damage_scalar.unwrap_or(0.1),
            duration_damage_scalar: self.duration_damage_scalar.unwrap_or(1.0),
        }
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct NodeFlavor {
    id: String,
    visible_name: String,
    description: String,
}

impl NodeFlavor {
    fn hydrate(self, index: usize) -> internal::NodeFlavor {
        internal::NodeFlavor {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct NodeTemplate {
    id: String,
    visible_name: String,
    description: String,
    visibility: bool,
    flavor: Vec<(String, u64)>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    factories: HashMap<String, (f32, u64, u64)>, //a list of the factories this node has, in the form of FactoryClass IDs
    shipyards: HashMap<String, (f32, u64, u64)>,
    environment: Vec<(String, u64)>, //name of the FRED environment to use for missions set in this node
    bitmap: Option<Vec<((String, f32), u64)>>,
    orphan: Option<bool>, //an orphaned node does not get all-to-all edges automatically built with the other nodes in its system
    allegiance: Vec<(String, u64)>, //faction that currently holds the node
    efficiency: Option<f32>, //efficiency of any production facilities in this node; changes over time based on faction ownership
    balance_stockpiles: Option<bool>,
    balance_hangars: Option<bool>,
    check_for_battles: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Node {
    id: String,
    template: String,
    visible_name: Option<String>, //location name as shown to player
    position: Option<[i64; 3]>, //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    description: Option<String>,
    visibility: Option<bool>,
    flavor: Option<String>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    factories: Option<Vec<String>>, //a list of the factories this node has, in the form of FactoryClass IDs
    shipyards: Option<Vec<String>>,
    environment: Option<String>, //name of the FRED environment to use for missions set in this node
    bitmap: Option<(String, f32)>,
    orphan: Option<bool>, //an orphaned node does not get all-to-all edges automatically built with the other nodes in its system
    allegiance: Option<String>, //faction that currently holds the node
    efficiency: Option<f32>, //efficiency of any production facilities in this node; changes over time based on faction ownership
    balance_stockpiles: Option<bool>,
    balance_hangars: Option<bool>,
    check_for_battles: Option<bool>,
}

impl Node {
    fn is_orphan(&self, nodetemplates: &Vec<NodeTemplate>) -> bool {
        match self.orphan {
            Some(val) => val,
            None => nodetemplates
                .iter()
                .find(|t| t.id == self.template)
                .unwrap()
                .clone()
                .orphan
                .unwrap_or(false),
        }
    }
    fn hydrate<R: Rng>(
        &self,
        index: usize,
        rng: &mut R,
        nodetemplates: &Vec<NodeTemplate>,
        nodeflavor_id_map: &HashMap<String, Arc<internal::NodeFlavor>>,
        factions: &HashMap<String, Arc<internal::Faction>>,
        factoryclass_id_map: &HashMap<String, Arc<internal::FactoryClass>>,
        shipyardclass_id_map: &HashMap<String, Arc<internal::ShipyardClass>>,
        shipclasses: &Vec<Arc<internal::ShipClass>>,
    ) -> internal::Node {
        let template = nodetemplates
            .iter()
            .find(|t| t.id == self.template)
            .unwrap()
            .clone();
        let faction = factions
            .get(&self.allegiance.clone().unwrap_or({
                template
                    .allegiance
                    .choose_weighted(rng, |(_, prob)| prob.clone())
                    .unwrap()
                    .0
                    .clone()
            }))
            .expect("Allegiance field is not correctly defined!")
            .clone();
        let node = internal::Node {
            id: index,
            visible_name: self
                .visible_name
                .clone()
                .unwrap_or(template.visible_name)
                .clone(),
            position: self.position.unwrap_or([0, 0, 0]),
            description: self
                .description
                .clone()
                .unwrap_or(template.description)
                .clone(),
            environment: self
                .environment
                .clone()
                .unwrap_or({
                    template
                        .environment
                        .choose_weighted(rng, |(_, prob)| prob.clone())
                        .unwrap()
                        .0
                        .clone()
                })
                .clone(),
            bitmap: match &self.bitmap {
                Some(bm) => Some(bm.clone()),
                None => template.bitmap.map(|bitmaps| {
                    bitmaps
                        .choose_weighted(rng, |(_, prob)| prob.clone())
                        .unwrap()
                        .0
                        .clone()
                }),
            },
            mutables: RwLock::new(internal::NodeMut {
                visibility: self.visibility.unwrap_or(template.visibility),
                flavor: nodeflavor_id_map
                    .get(&self.flavor.clone().unwrap_or({
                        template
                            .flavor
                            .choose_weighted(rng, |(_, prob)| prob.clone())
                            .unwrap()
                            .0
                            .clone()
                    }))
                    .expect("Node flavor field is not correctly defined!")
                    .clone(),
                factories: match &self.factories {
                    Some(list) => list
                        .iter()
                        .map(|stringid| {
                            internal::FactoryClass::instantiate(
                                factoryclass_id_map
                                    .get(stringid)
                                    .expect(&format!(
                                        "Factory class '{}' does not exist.",
                                        stringid
                                    ))
                                    .clone(),
                            )
                        })
                        .collect(),
                    None => template
                        .factories
                        .iter()
                        .map(|(stringid, (factor, min, max))| {
                            let attempt_range = (*min..=*max).collect::<Vec<_>>();
                            let attempt_count = attempt_range.choose(rng).unwrap();
                            (0..=*attempt_count)
                                .collect::<Vec<_>>()
                                .iter()
                                .filter(|_| rng.gen_range(0.0..1.0) < *factor)
                                .map(|_| {
                                    internal::FactoryClass::instantiate(
                                        factoryclass_id_map
                                            .get(stringid)
                                            .expect(&format!(
                                                "Factory class '{}' does not exist.",
                                                stringid
                                            ))
                                            .clone(),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .flatten()
                        .collect(),
                },
                shipyards: match &self.shipyards {
                    Some(list) => list
                        .iter()
                        .map(|stringid| {
                            internal::ShipyardClass::instantiate(
                                shipyardclass_id_map
                                    .get(stringid)
                                    .expect(&format!(
                                        "Shipyard class '{}' does not exist.",
                                        stringid
                                    ))
                                    .clone(),
                                shipclasses,
                            )
                        })
                        .collect(),
                    None => template
                        .shipyards
                        .iter()
                        .map(|(stringid, (factor, min, max))| {
                            let attempt_range = (*min..=*max).collect::<Vec<_>>();
                            let attempt_count = attempt_range.choose(rng).unwrap();
                            (0..=*attempt_count)
                                .collect::<Vec<_>>()
                                .iter()
                                .filter(|_| rng.gen_range(0.0..1.0) < *factor)
                                .map(|_| {
                                    internal::ShipyardClass::instantiate(
                                        shipyardclass_id_map
                                            .get(stringid)
                                            .expect(&format!(
                                                "Factory class '{}' does not exist.",
                                                stringid
                                            ))
                                            .clone(),
                                        shipclasses,
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .flatten()
                        .collect(),
                },
                allegiance: faction.clone(),
                efficiency: self
                    .efficiency
                    .unwrap_or(template.efficiency.unwrap_or(faction.efficiency_default)),
                transact_resources: self
                    .balance_stockpiles
                    .unwrap_or(template.balance_stockpiles.unwrap_or(true)),
                transact_units: self
                    .balance_hangars
                    .unwrap_or(template.balance_hangars.unwrap_or(true)),
                check_for_battles: self
                    .check_for_battles
                    .unwrap_or(template.check_for_battles.unwrap_or(true)),
                resources_transacted: false,
                units_transacted: false,
            }),
            unit_container: RwLock::new(internal::UnitContainer {
                contents: Vec::new(),
            }),
        };
        node
    }
}

#[derive(Serialize, Deserialize)]
struct System {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    nodes: Vec<Node>,
}

impl System {
    fn hydrate(
        self,
        index: usize,
        node_id_map: &HashMap<String, Arc<internal::Node>>,
    ) -> internal::System {
        let internalsystem = internal::System {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            nodes: self
                .nodes
                .iter()
                .map(|node| node_id_map.get(&node.id).unwrap().clone())
                .collect(),
        };
        internalsystem
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct EdgeFlavor {
    id: String,
    visible_name: String,
    description: String,
    propagates: bool,
}

impl EdgeFlavor {
    fn hydrate(self, index: usize) -> internal::EdgeFlavor {
        internal::EdgeFlavor {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            propagates: self.propagates,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Faction {
    id: String,
    visible_name: String, //faction name as shown to player
    description: String,
    visibility: Option<bool>,
    propagates: Option<bool>,
    efficiency_default: f32, //starting value for production facility efficiency
    efficiency_target: f32, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    efficiency_delta: f32,  //rate at which efficiency changes
    battle_scalar: Option<f32>,
    value_mult: Option<f32>,
    volume_strength_ratio: Option<f32>,
    relations: HashMap<String, f32>,
}

impl Faction {
    fn hydrate(
        self,
        index: usize,
        faction_id_map: &HashMap<String, internal::FactionID>,
    ) -> internal::Faction {
        let faction = internal::Faction {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            efficiency_default: self.efficiency_default,
            efficiency_target: self.efficiency_target,
            efficiency_delta: self.efficiency_delta,
            battle_scalar: self.battle_scalar.unwrap_or(1.0),
            value_mult: self.value_mult.clone().unwrap_or(1.0),
            volume_strength_ratio: self.volume_strength_ratio.unwrap_or(1.0),
            relations: faction_id_map
                .iter()
                .map(|(name, id)| (*id, self.relations.get(name).map(|&x| x).unwrap_or(0_f32)))
                .collect(),
        };
        faction
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Resource {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    propagates: Option<bool>,
    unit_vol: u64, //how much space a one unit of this resource takes up when transported by a cargo ship
    value_mult: Option<f32>, //how valuable the AI considers one unit of this resource to be
}

impl Resource {
    fn hydrate(self, index: usize) -> internal::Resource {
        let resource = internal::Resource {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            unit_vol: self.unit_vol,
            value_mult: self.value_mult.clone().unwrap_or(1.0),
        };
        resource
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnipotentStockpile {
    visibility: Option<bool>,
    resource_type: String,
    contents: u64,
    rate: Option<u64>,
    target: u64,
    capacity: u64,
    propagates: Option<bool>,
}

impl UnipotentStockpile {
    fn hydrate(
        self,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::UnipotentStockpile {
        let stockpile = internal::UnipotentStockpile {
            visibility: self.visibility.unwrap_or(true),
            resource_type: resource_id_map
                .get(&self.resource_type)
                .expect("Resource is invalid!")
                .clone(),
            contents: self.contents,
            rate: self.rate.unwrap_or(0),
            target: self.target,
            capacity: self.capacity,
            propagates: self.propagates.unwrap_or(true),
        };
        stockpile
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluripotentStockpile {
    visibility: Option<bool>,
    contents: HashMap<String, u64>,
    allowed: Option<Vec<String>>,
    target: u64,
    capacity: u64,
    propagates: Option<bool>,
}

impl PluripotentStockpile {
    fn hydrate(
        mut self,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::PluripotentStockpile {
        let stockpile = internal::PluripotentStockpile {
            visibility: self.visibility.unwrap_or(true),
            contents: self
                .contents
                .drain()
                .map(|(id, num)| {
                    (
                        resource_id_map
                            .get(&id)
                            .expect("Resource is invalid!")
                            .clone(),
                        num,
                    )
                })
                .collect(),
            allowed: self.allowed.map(|resources| {
                resources
                    .iter()
                    .map(|id| {
                        resource_id_map
                            .get(id)
                            .expect("Resource is invalid!")
                            .clone()
                    })
                    .collect()
            }),
            target: self.target,
            capacity: self.capacity,
            propagates: self.propagates.unwrap_or(true),
        };
        stockpile
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct HangarClass {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    capacity: u64,                         //total volume the hangar can hold
    target: u64, //volume the hangar wants to hold; this is usually either equal to capacity (for carriers) or zero (for shipyard outputs)
    allowed: Vec<String>, //which shipclasses this hangar can hold
    ideal: HashMap<String, u64>, //how many of each ship type the hangar wants
    sub_target_supply_scalar: Option<f32>, //multiplier used for supply generated by non-ideal units under the target limit; should be below one
    non_ideal_demand_scalar: Option<f32>, //multiplier used for demand generated for non-ideal unitclasses; should be below one
    transport: Option<bool>,
    launch_volume: u64, //how much volume the hangar can launch at one time in battle
    launch_interval: u64, //time between launches in battle
    propagates: Option<bool>, //whether or not hangar generates saliences
}

impl HangarClass {
    fn hydrate(
        mut self,
        index: usize,
        shipclass_id_map: &HashMap<String, internal::ShipClassID>,
        squadronclass_id_map: &HashMap<String, internal::SquadronClassID>,
    ) -> internal::HangarClass {
        let hangarclass = internal::HangarClass {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            capacity: self.capacity,
            target: self.target,
            allowed: self
                .allowed
                .drain(0..)
                .map(|x| {
                    if let Some(shipclassid) = shipclass_id_map.get(&x) {
                        internal::UnitClassID::ShipClass(*shipclassid)
                    } else {
                        internal::UnitClassID::SquadronClass(*squadronclass_id_map.get(&x).unwrap())
                    }
                })
                .collect(),
            ideal: self
                .ideal
                .drain()
                .map(|(k, v)| {
                    (
                        {
                            if let Some(shipclassid) = shipclass_id_map.get(&k) {
                                internal::UnitClassID::ShipClass(*shipclassid)
                            } else {
                                internal::UnitClassID::SquadronClass(
                                    *squadronclass_id_map.get(&k).unwrap(),
                                )
                            }
                        },
                        v,
                    )
                })
                .collect(),
            sub_target_supply_scalar: self.sub_target_supply_scalar.unwrap_or(0.5),
            non_ideal_demand_scalar: self.non_ideal_demand_scalar.unwrap_or(0.5),
            transport: self.transport.unwrap_or(false),
            launch_volume: self.launch_volume,
            launch_interval: self.launch_interval,
            propagates: self.propagates.unwrap_or(true),
        };
        hangarclass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EngineClass {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    base_health: Option<u64>,
    toughness_scalar: f32,
    inputs: Vec<UnipotentStockpile>,
    forbidden_nodeflavors: Option<Vec<String>>,
    forbidden_edgeflavors: Option<Vec<String>>,
    speed: u64,    //number of edges the engine allows a ship to traverse when used
    cooldown: u64, //number of turns engine must wait before being used again
}

impl EngineClass {
    fn hydrate(
        mut self,
        index: usize,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
        nodeflavor_id_map: &HashMap<String, Arc<internal::NodeFlavor>>,
        edgeflavor_id_map: &HashMap<String, Arc<internal::EdgeFlavor>>,
    ) -> internal::EngineClass {
        internal::EngineClass {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            base_health: self.base_health,
            toughness_scalar: self.toughness_scalar,
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resource_id_map))
                .collect(),
            forbidden_nodeflavors: match self.forbidden_nodeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| nodeflavor_id_map.get(&s).unwrap().clone())
                    .collect(),
                None => Vec::new(),
            },
            forbidden_edgeflavors: match self.forbidden_edgeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| edgeflavor_id_map.get(&s).unwrap().clone())
                    .collect(),
                None => Vec::new(),
            },
            speed: self.speed,
            cooldown: self.cooldown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RepairerClass {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>,
    repair_points: i64,
    repair_factor: f32,
    engine_repair_points: i64,
    engine_repair_factor: f32,
    strategic_weapon_repair_points: i64,
    strategic_weapon_repair_factor: f32,
    per_engagement: Option<bool>, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerClass {
    fn hydrate(
        mut self,
        index: usize,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::RepairerClass {
        let repairerclass = internal::RepairerClass {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resource_id_map))
                .collect(),
            repair_points: self.repair_points,
            repair_factor: self.repair_factor,
            engine_repair_points: self.engine_repair_points,
            engine_repair_factor: self.engine_repair_factor,
            subsystem_repair_points: self.strategic_weapon_repair_points,
            subsystem_repair_factor: self.strategic_weapon_repair_factor,
            per_engagement: self.per_engagement.unwrap_or(false),
        };
        repairerclass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StrategicWeaponClass {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>,
    forbidden_nodeflavors: Option<Vec<String>>, //the weapon won't fire into nodes of these flavors
    forbidden_edgeflavors: Option<Vec<String>>, //the weapon won't fire across edges of these flavors
    damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage done by a single shot
    engine_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to engine done by a single shot
    strategic_weapon_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to strategic weapon done by a single shot
    accuracy: f32, //divided by target's strategicweaponevasionscalar to get hit probability as a fraction of 1.0
    range: u64,    //how many edges away the weapon can reach
    shots: (u64, u64), //lower and upper bounds for maximum number of shots the weapon fires each time it's activated
    targets_enemies: bool,
    targets_allies: bool,
    targets_neutrals: bool,
    target_relations_lower_bound: Option<f32>,
    target_relations_upper_bound: Option<f32>,
    target_priorities_class: Option<HashMap<String, f32>>, //how strongly weapon will prioritize ships of each class; classes absent from list will default to 1.0
    target_priorities_flavor: Option<HashMap<String, f32>>, //how strongly weapon will prioritize ships of each flavor; flavors absent from list will default to 1.0
}

impl StrategicWeaponClass {
    fn hydrate(
        mut self,
        index: usize,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
        nodeflavor_id_map: &HashMap<String, Arc<internal::NodeFlavor>>,
        edgeflavor_id_map: &HashMap<String, Arc<internal::EdgeFlavor>>,
        shipflavor_id_map: &HashMap<String, Arc<internal::ShipFlavor>>,
        shipclass_id_map: &HashMap<String, internal::ShipClassID>,
    ) -> internal::StrategicWeaponClass {
        internal::StrategicWeaponClass {
            id: index,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resource_id_map))
                .collect(),
            forbidden_nodeflavors: match self.forbidden_nodeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| nodeflavor_id_map.get(&s).unwrap().clone())
                    .collect(),
                None => Vec::new(),
            },
            forbidden_edgeflavors: match self.forbidden_edgeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| edgeflavor_id_map.get(&s).unwrap().clone())
                    .collect(),
                None => Vec::new(),
            },
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
            target_priorities_class: match self.target_priorities_class {
                Some(mut v) => v
                    .drain()
                    .map(|(class, val)| (shipclass_id_map.get(&class).unwrap().clone(), val))
                    .collect(),
                None => HashMap::new(),
            },
            target_priorities_flavor: match self.target_priorities_flavor {
                Some(mut v) => v
                    .drain()
                    .map(|(flavor, val)| (shipflavor_id_map.get(&flavor).unwrap().clone(), val))
                    .collect(),
                None => HashMap::new(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FactoryClass {
    id: String,
    visible_name: String,
    description: String,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    fn hydrate(
        mut self,
        index: usize,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::FactoryClass {
        let factoryclass = internal::FactoryClass {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resource_id_map))
                .collect(),
            outputs: self
                .outputs
                .drain(0..)
                .map(|x| x.hydrate(resource_id_map))
                .collect(),
        };
        factoryclass
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShipyardClass {
    id: String,
    visible_name: Option<String>,
    description: Option<String>,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>,
    outputs: HashMap<String, u64>,
    construct_rate: u64,
    efficiency: f32,
}

impl ShipyardClass {
    fn hydrate(
        mut self,
        index: usize,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
        shipclass_id_map: &HashMap<String, internal::ShipClassID>,
    ) -> internal::ShipyardClass {
        let shipyardclass = internal::ShipyardClass {
            id: index,
            visible_name: self.visible_name,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resource_id_map))
                .collect(),
            outputs: self
                .outputs
                .drain()
                .map(|(k, v)| (*shipclass_id_map.get(&k).unwrap(), v))
                .collect(),
            construct_rate: self.construct_rate,
            efficiency: self.efficiency,
        };
        shipyardclass
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubsystemClass {
    id: String,
    visible_name: String,
    visibility: Option<bool>,
    base_health: Option<u64>,
    toughness_scalar: Option<f32>,
    strength_mod: (f32, u64),
}

impl SubsystemClass {
    fn hydrate(self, index: usize) -> internal::SubsystemClass {
        internal::SubsystemClass {
            id: index,
            visible_name: self.visible_name,
            visibility: self.visibility.unwrap_or(true),
            base_health: self.base_health,
            toughness_scalar: self.toughness_scalar.unwrap_or(1.0),
            strength_mod: self.strength_mod,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShipAI {
    id: String,
    nav_threshold: Option<f32>,
    ship_attract_specific: f32, //a multiplier for supply gradients corresponding to the specific class of a ship using this AI
    ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship supply gradients
    ship_cargo_attract: HashMap<String, f32>, //a list of ship classes whose supply gradients this AI will follow (so as to carry e.g. fighters that can't travel on their own), and individual strength multipliers
    resource_attract: HashMap<String, f32>, //a list of resources whose supply gradients this AI will follow, and individual strength multipliers
    friendly_supply_attract: f32,
    hostile_supply_attract: f32,
    allegiance_demand_attract: f32,
    enemy_demand_attract: f32,
    strategic_weapon_damage_attract: f32,
    strategic_weapon_engine_damage_attract: f32,
    strategic_weapon_subsystem_damage_attract: f32,
    strategic_weapon_healing_attract: f32,
    strategic_weapon_engine_healing_attract: f32,
    strategic_weapon_subsystem_healing_attract: f32,
}

impl ShipAI {
    fn hydrate(
        self,
        index: usize,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
        shipclass_id_map: &HashMap<String, internal::ShipClassID>,
    ) -> internal::ShipAI {
        let shipai = internal::ShipAI {
            id: index,
            nav_threshold: self.nav_threshold.unwrap_or(1.0),
            ship_attract_specific: self.ship_attract_specific,
            ship_attract_generic: self.ship_attract_generic,
            ship_cargo_attract: self
                .ship_cargo_attract
                .iter()
                .map(|(stringid, v)| {
                    (
                        internal::UnitClassID::ShipClass(*shipclass_id_map.get(stringid).unwrap()),
                        *v,
                    )
                })
                .collect(),
            resource_attract: self
                .resource_attract
                .iter()
                .map(|(stringid, v)| (resource_id_map.get(stringid).unwrap().clone(), *v))
                .collect(),
            friendly_supply_attract: self.friendly_supply_attract.clone(),
            hostile_supply_attract: self.hostile_supply_attract.clone(),
            allegiance_demand_attract: self.allegiance_demand_attract.clone(),
            enemy_demand_attract: self.enemy_demand_attract.clone(),
            strategic_weapon_damage_attract: self.strategic_weapon_damage_attract,
            strategic_weapon_engine_damage_attract: self.strategic_weapon_engine_damage_attract,
            strategic_weapon_subsystem_damage_attract: self
                .strategic_weapon_subsystem_damage_attract,
            strategic_weapon_healing_attract: self.strategic_weapon_healing_attract,
            strategic_weapon_engine_healing_attract: self.strategic_weapon_engine_healing_attract,
            strategic_weapon_subsystem_healing_attract: self
                .strategic_weapon_subsystem_healing_attract,
        };
        shipai
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipFlavor {
    pub id: String,
    pub visible_name: String,
    pub description: String,
}

impl ShipFlavor {
    fn hydrate(self, index: usize) -> internal::ShipFlavor {
        internal::ShipFlavor {
            id: index,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ShipClass {
    id: String,
    visible_name: String,
    description: String,
    shipflavor: String,
    base_hull: u64,     //how many hull hitpoints this ship has by default
    base_strength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    visibility: Option<bool>,
    propagates: Option<bool>,
    hangar_vol: u64, //how much hangar space this ship takes up when carried by a host
    stockpiles: Option<Vec<PluripotentStockpile>>,
    default_weapons: Option<HashMap<String, u64>>, //a strikecraft's default weapons, which it always has with it
    hangars: Option<Vec<String>>,
    engines: Option<Vec<String>>,
    repairers: Option<Vec<String>>,
    strategic_weapons: Option<Vec<String>>,
    factories: Option<Vec<String>>,
    shipyards: Option<Vec<String>>,
    subsystems: Option<Vec<String>>,
    ai_class: String,
    processor_demand_nav_scalar: Option<f32>,
    deploys_self: Option<bool>,
    deploys_daughters: Option<u64>,
    mother_misalignment_tolerance: Option<f32>,
    defect_chance: Option<HashMap<String, (f32, f32)>>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    toughness_scalar: Option<f32>,
    battle_escape_scalar: Option<f32>,
    defect_escape_scalar: Option<f32>,
    interdiction_scalar: Option<f32>,
    strategic_weapon_evasion_scalar: Option<f32>,
    value_mult: Option<f32>, //how valuable the AI considers one volume point of this shipclass to be
}

impl ShipClass {
    fn hydrate(
        &self,
        shipclass_id_map: &HashMap<String, internal::ShipClassID>,
        shipflavor_id_map: &HashMap<String, Arc<internal::ShipFlavor>>,
        resource_id_map: &HashMap<String, Arc<internal::Resource>>,
        hangarclass_id_map: &HashMap<String, Arc<internal::HangarClass>>,
        engineclass_id_map: &HashMap<String, Arc<internal::EngineClass>>,
        repairerclass_id_map: &HashMap<String, Arc<internal::RepairerClass>>,
        strategicweaponclass_id_map: &HashMap<String, Arc<internal::StrategicWeaponClass>>,
        factoryclass_id_map: &HashMap<String, Arc<internal::FactoryClass>>,
        shipyardclass_id_map: &HashMap<String, Arc<internal::ShipyardClass>>,
        subsystemclass_id_map: &HashMap<String, Arc<internal::SubsystemClass>>,
        shipai_id_map: &HashMap<String, Arc<internal::ShipAI>>,
        factions: &HashMap<String, Arc<internal::Faction>>,
    ) -> internal::ShipClass {
        let shipclass = internal::ShipClass {
            id: shipclass_id_map.get(&self.id).unwrap().index,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            shipflavor: shipflavor_id_map.get(&self.shipflavor).unwrap().clone(),
            base_hull: self.base_hull,
            base_strength: self.base_strength,
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            default_weapons: self.default_weapons.clone().map(|map| {
                map.iter()
                    .map(|(id, n)| {
                        (
                            resource_id_map
                                .get(id)
                                .unwrap_or_else(|| panic!("{} is not found!", id))
                                .clone(),
                            *n,
                        )
                    })
                    .collect()
            }),
            hangar_vol: self.hangar_vol,
            stockpiles: self
                .stockpiles
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|stockpile| stockpile.clone().hydrate(&resource_id_map))
                .collect(),
            hangars: self
                .hangars
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    hangarclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            engines: self
                .engines
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    engineclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            repairers: self
                .repairers
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    repairerclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            strategic_weapons: self
                .strategic_weapons
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    strategicweaponclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            factories: self
                .factories
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    factoryclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            shipyards: self
                .shipyards
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    shipyardclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            subsystems: self
                .subsystems
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    subsystemclass_id_map
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            ai_class: shipai_id_map.get(&self.ai_class).unwrap().clone(),
            processor_demand_nav_scalar: self.processor_demand_nav_scalar.unwrap_or(10.0),
            deploys_self: self.deploys_self.unwrap_or(true),
            deploys_daughters: self.deploys_daughters,
            mother_misalignment_tolerance: self.mother_misalignment_tolerance,
            defect_chance: self
                .defect_chance
                .clone()
                .unwrap_or(HashMap::new())
                .iter()
                .map(|(k, v)| (factions.get(k).unwrap().clone(), *v))
                .collect(),
            toughness_scalar: self.toughness_scalar.unwrap_or(1.0),
            battle_escape_scalar: self.battle_escape_scalar.unwrap_or(1.0),
            defect_escape_scalar: self.defect_escape_scalar.unwrap_or(1.0),
            interdiction_scalar: self.interdiction_scalar.unwrap_or(1.0),
            strategic_weapon_evasion_scalar: self.strategic_weapon_evasion_scalar.unwrap_or(1.0),
            value_mult: self.value_mult.clone().unwrap_or(1.0),
        };
        shipclass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronFlavor {
    pub id: String,
    pub visible_name: String,
    pub description: String,
}

impl SquadronFlavor {
    fn hydrate(self, index: usize) -> internal::SquadronFlavor {
        internal::SquadronFlavor {
            id: index,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SquadronClass {
    id: String,
    visible_name: String,
    description: String,
    squadron_flavor: String,
    visibility: Option<bool>,
    capacity: Option<u64>,
    target: Option<u64>,
    propagates: Option<bool>,
    strength_mod: (f32, u64),
    squadron_config: HashMap<String, u64>,
    sub_target_supply_scalar: Option<f32>, //multiplier used for demand generated by non-ideal ships under the target limit; should be below one
    non_ideal_demand_scalar: Option<f32>, //multiplier used for demand generated for non-ideal unitclasses; should be below one
    nav_quorum: f32,
    disband_threshold: f32,
    deploys_self: Option<bool>,
    deploys_daughters: Option<Option<u64>>,
    defect_chance: Option<HashMap<String, (f32, f32)>>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    defect_escape_mod: Option<f32>,
    value_mult: Option<f32>, //how valuable the AI considers one volume point of this squadronclass to be
}

impl SquadronClass {
    fn get_target(
        &self,
        shipclasses: &Vec<ShipClass>,
        squadronclasses: &Vec<SquadronClass>,
    ) -> u64 {
        self.squadron_config
            .iter()
            .map(|(stringid, _)| {
                if let Some(shipclass) = shipclasses.iter().find(|sc| sc.id == *stringid) {
                    shipclass.hangar_vol
                } else {
                    squadronclasses
                        .iter()
                        .find(|fc| fc.id == *stringid)
                        .unwrap()
                        .get_target(shipclasses, squadronclasses)
                }
            })
            .sum()
    }
    fn hydrate(
        &self,
        index: usize,
        shipclass_id_map: &HashMap<String, internal::ShipClassID>,
        squadronclass_id_map: &HashMap<String, internal::SquadronClassID>,
        squadronflavor_id_map: &HashMap<String, Arc<internal::SquadronFlavor>>,
        factions: &HashMap<String, Arc<internal::Faction>>,
        shipclasses: &Vec<ShipClass>,
        squadronclasses: &Vec<SquadronClass>,
    ) -> internal::SquadronClass {
        let squadronclass = internal::SquadronClass {
            id: index,
            visible_name: self.visible_name.clone(),
            description: self.description.clone(),
            squadronflavor: squadronflavor_id_map
                .get(&self.squadron_flavor)
                .unwrap()
                .clone(),
            visibility: self.visibility.unwrap_or(true),
            capacity: self
                .capacity
                .unwrap_or(self.get_target(shipclasses, squadronclasses)),
            target: self
                .target
                .unwrap_or(self.get_target(shipclasses, squadronclasses)),
            propagates: self.propagates.unwrap_or(true),
            strength_mod: self.strength_mod.clone(),
            squadron_config: self
                .squadron_config
                .iter()
                .map(|(stringid, n)| {
                    (
                        {
                            if let Some(shipclassid) = shipclass_id_map.get(&stringid.clone()) {
                                internal::UnitClassID::ShipClass(*shipclassid)
                            } else {
                                internal::UnitClassID::SquadronClass(
                                    *squadronclass_id_map.get(&stringid.clone()).unwrap(),
                                )
                            }
                        },
                        *n,
                    )
                })
                .collect(),
            sub_target_supply_scalar: self.sub_target_supply_scalar.unwrap_or(0.5),
            non_ideal_demand_scalar: self.non_ideal_demand_scalar.unwrap_or(0.5),
            nav_quorum: self.nav_quorum,
            disband_threshold: self.disband_threshold,
            deploys_self: self.deploys_self.unwrap_or(true),
            deploys_daughters: self.deploys_daughters.unwrap_or(None),
            defect_chance: self
                .defect_chance
                .clone()
                .unwrap_or(HashMap::new())
                .iter()
                .map(|(stringid, n)| (factions.get(stringid).unwrap().clone(), *n))
                .collect(),
            defect_escape_mod: self.defect_escape_mod.unwrap_or(1.0),
            value_mult: self.value_mult.clone().unwrap_or(1.0),
        };
        squadronclass
    }
}

#[derive(Serialize, Deserialize)] //structure for modder-defined json
pub struct Root {
    config: Config,
    nodeflavors: Vec<NodeFlavor>,
    nodetemplates: Vec<NodeTemplate>,
    systems: Vec<System>,
    edgeflavors: Vec<EdgeFlavor>,
    pure_internal_edgeflavor: String,
    semi_internal_edgeflavor: String,
    edges: Vec<(String, String, String)>,
    factions: Vec<Faction>,
    wars: Vec<(String, String)>,
    resources: Vec<Resource>,
    hangarclasses: Vec<HangarClass>,
    engineclasses: Vec<EngineClass>,
    repairerclasses: Vec<RepairerClass>,
    strategicweaponclasses: Vec<StrategicWeaponClass>,
    factoryclasses: Vec<FactoryClass>,
    shipyardclasses: Vec<ShipyardClass>,
    subsystemclasses: Vec<SubsystemClass>,
    shipais: Vec<ShipAI>,
    shipflavors: Vec<ShipFlavor>,
    squadronflavors: Vec<SquadronFlavor>,
    shipclasses: Vec<ShipClass>,
    squadronclasses: Vec<SquadronClass>,
}

impl Root {
    //hydration method
    pub fn hydrate(mut self) -> internal::Root {
        let unitclasscounter = Arc::new(AtomicUsize::new(0));

        let config = self.config.hydrate();

        let nodeflavor_id_map: HashMap<String, Arc<internal::NodeFlavor>> = self
            .nodeflavors
            .drain(0..)
            .enumerate()
            .map(|(i, nodeflavor)| (nodeflavor.id.clone(), Arc::new(nodeflavor.hydrate(i))))
            .collect();

        let edgeflavor_id_map: HashMap<String, Arc<internal::EdgeFlavor>> = self
            .edgeflavors
            .drain(0..)
            .enumerate()
            .map(|(i, edgeflavor)| (edgeflavor.id.clone(), Arc::new(edgeflavor.hydrate(i))))
            .collect();

        let faction_id_map: HashMap<String, internal::FactionID> = self
            .factions
            .iter()
            .enumerate()
            .map(|(i, faction)| {
                let stringid = faction.id.clone();
                let kv_pair = (stringid, internal::FactionID::new_from_index(i));
                kv_pair
            })
            .collect();

        //fairly simple hydration process
        let factions: HashMap<String, Arc<internal::Faction>> = self
            .factions
            .drain(0..)
            .enumerate()
            .map(|(i, faction)| {
                //we make sure the enumeration index we have matches the faction's entry in the _id_map
                assert_eq!(i, faction_id_map.get(&faction.id).unwrap().index);
                (
                    faction.id.clone(),
                    Arc::new(faction.hydrate(i, &faction_id_map)),
                )
            })
            .collect();

        let wars: HashSet<(Arc<internal::Faction>, Arc<internal::Faction>)> = self
            .wars
            .iter()
            .map(|(a, b)| {
                let aid = factions.get(a).unwrap().clone();
                let bid = factions.get(b).unwrap().clone();
                assert_ne!(aid, bid);
                (aid.clone().min(bid.clone()), bid.clone().max(aid.clone()))
            })
            .collect();

        //same sort of deal here
        let resource_id_map: HashMap<String, Arc<internal::Resource>> = self
            .resources
            .drain(0..)
            .enumerate()
            .map(|(i, resource)| (resource.id.clone(), Arc::new(resource.hydrate(i))))
            .collect();

        let engineclass_id_map: HashMap<String, Arc<internal::EngineClass>> = self
            .engineclasses
            .drain(0..)
            .enumerate()
            .map(|(i, engineclass)| {
                (
                    engineclass.id.clone(),
                    Arc::new(engineclass.hydrate(
                        i,
                        &resource_id_map,
                        &nodeflavor_id_map,
                        &edgeflavor_id_map,
                    )),
                )
            })
            .collect();

        let repairerclass_id_map: HashMap<String, Arc<internal::RepairerClass>> = self
            .repairerclasses
            .drain(0..)
            .enumerate()
            .map(|(i, repairerclass)| {
                (
                    repairerclass.id.clone(),
                    Arc::new(repairerclass.hydrate(i, &resource_id_map)),
                )
            })
            .collect();

        let factoryclass_id_map: HashMap<String, Arc<internal::FactoryClass>> = self
            .factoryclasses
            .drain(0..)
            .enumerate()
            .map(|(i, factoryclass)| {
                (
                    factoryclass.id.clone(),
                    Arc::new(factoryclass.hydrate(i, &resource_id_map)),
                )
            })
            .collect();

        let subsystemclass_id_map: HashMap<String, Arc<internal::SubsystemClass>> = self
            .subsystemclasses
            .drain(0..)
            .enumerate()
            .map(|(i, subsystemclass)| {
                (
                    subsystemclass.id.clone(),
                    Arc::new(subsystemclass.hydrate(i)),
                )
            })
            .collect();

        //this is a dummy ship class, which is here so that salience processes that require a shipclass to be specified can be parsed correctly
        let generic_demand_ship = ShipClass {
            id: "generic_demand_ship".to_string(),
            visible_name: "Generic Demand Ship".to_string(),
            description: "".to_string(),
            shipflavor: "default".to_string(),
            base_hull: 1,     //how many hull hitpoints this ship has by default
            base_strength: 0, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
            visibility: Some(false),
            propagates: Some(false),
            hangar_vol: 0, //how much hangar space this ship takes up when carried by a host
            stockpiles: None,
            default_weapons: None, //a strikecraft's default weapons, which it always has with it
            hangars: None,
            engines: None,
            repairers: None,
            strategic_weapons: None,
            factories: None,
            shipyards: None,
            subsystems: None,
            ai_class: "default".to_string(), //aiclass
            processor_demand_nav_scalar: None,
            deploys_self: None,
            deploys_daughters: None,
            mother_misalignment_tolerance: None,
            defect_chance: None,
            toughness_scalar: None,
            battle_escape_scalar: None,
            defect_escape_scalar: None,
            interdiction_scalar: None,
            strategic_weapon_evasion_scalar: None,
            value_mult: None,
        };

        //here we create the shipclass_id_map, put the dummy ship class inside it, and then insert all the actual ship classes
        let shipclass_id_map: HashMap<String, internal::ShipClassID> =
            iter::once(&generic_demand_ship)
                .chain(self.shipclasses.iter())
                .map(|shipclass| {
                    (
                        shipclass.id.clone(),
                        internal::ShipClassID::new_from_index(
                            unitclasscounter.fetch_add(1, atomic::Ordering::Relaxed),
                        ),
                    )
                })
                .collect();

        let squadronclass_id_map: HashMap<String, internal::SquadronClassID> = self
            .squadronclasses
            .iter()
            .map(|squadronclass| {
                (
                    squadronclass.id.clone(),
                    internal::SquadronClassID::new_from_index(
                        unitclasscounter.fetch_add(1, atomic::Ordering::Relaxed),
                    ),
                )
            })
            .collect();

        let hangarclass_id_map: HashMap<String, Arc<internal::HangarClass>> = self
            .hangarclasses
            .drain(0..)
            .enumerate()
            .map(|(i, hangarclass)| {
                (
                    hangarclass.id.clone(),
                    Arc::new(hangarclass.hydrate(i, &shipclass_id_map, &squadronclass_id_map)),
                )
            })
            .collect();

        let shipyardclass_id_map: HashMap<String, Arc<internal::ShipyardClass>> = self
            .shipyardclasses
            .drain(0..)
            .enumerate()
            .map(|(i, shipyardclass)| {
                (
                    shipyardclass.id.clone(),
                    Arc::new(shipyardclass.hydrate(i, &resource_id_map, &shipclass_id_map)),
                )
            })
            .collect();

        let shipai_id_map: HashMap<String, Arc<internal::ShipAI>> = self
            .shipais
            .drain(0..)
            .enumerate()
            .map(|(i, shipai)| {
                (
                    shipai.id.clone(),
                    Arc::new(shipai.hydrate(i, &resource_id_map, &shipclass_id_map)),
                )
            })
            .collect();

        let shipflavor_id_map: HashMap<String, Arc<internal::ShipFlavor>> = self
            .shipflavors
            .drain(0..)
            .enumerate()
            .map(|(i, shipflavor)| (shipflavor.id.clone(), Arc::new(shipflavor.hydrate(i))))
            .collect();

        let strategicweaponclass_id_map: HashMap<String, Arc<internal::StrategicWeaponClass>> =
            self.strategicweaponclasses
                .drain(0..)
                .enumerate()
                .map(|(i, strategicweaponclass)| {
                    (
                        strategicweaponclass.id.clone(),
                        Arc::new(strategicweaponclass.hydrate(
                            i,
                            &resource_id_map,
                            &nodeflavor_id_map,
                            &edgeflavor_id_map,
                            &shipflavor_id_map,
                            &shipclass_id_map,
                        )),
                    )
                })
                .collect();

        //we hydrate shipclasses, starting with the generic demand ship
        let shipclasses: Vec<Arc<internal::ShipClass>> = iter::once(&generic_demand_ship)
            .chain(self.shipclasses.iter())
            .map(|shipclass| {
                Arc::new(shipclass.hydrate(
                    &shipclass_id_map,
                    &shipflavor_id_map,
                    &resource_id_map,
                    &hangarclass_id_map,
                    &engineclass_id_map,
                    &repairerclass_id_map,
                    &strategicweaponclass_id_map,
                    &factoryclass_id_map,
                    &shipyardclass_id_map,
                    &subsystemclass_id_map,
                    &shipai_id_map,
                    &factions,
                ))
            })
            .collect();

        let mut rng = rand_hc::Hc128Rng::seed_from_u64(1138);

        //here we iterate over the json systems to create a map between nodes' json string-ids and internal ids
        let node_id_map: HashMap<String, Arc<internal::Node>> = self
            .systems
            .iter()
            .flat_map(|system| system.nodes.iter())
            .enumerate()
            .map(|(i, node)| {
                let nodehydration = node.hydrate(
                    i,
                    &mut rng,
                    &self.nodetemplates,
                    &nodeflavor_id_map,
                    &factions,
                    &factoryclass_id_map,
                    &shipyardclass_id_map,
                    &shipclasses,
                );
                (node.id.clone(), Arc::new(nodehydration))
            })
            .collect();

        //here we convert the json edge list into a set of pairs of internal node ids
        let mut edges: HashMap<
            (Arc<internal::Node>, Arc<internal::Node>),
            Arc<internal::EdgeFlavor>,
        > = self
            .edges
            .iter()
            .map(|(a, b, f)| {
                let aid = node_id_map.get(a).unwrap().clone();
                let bid = node_id_map.get(b).unwrap().clone();
                assert_ne!(aid, bid);
                (
                    (aid.clone().min(bid.clone()), bid.max(aid)),
                    edgeflavor_id_map.get(f).unwrap().clone(),
                )
            })
            .collect();

        let system_id_map: HashMap<String, Arc<internal::System>> = self
            .systems
            .drain(0..)
            .enumerate()
            .map(|(i, system)| {
                let mut nodestringids: Vec<String> = Vec::new();
                for node in &system.nodes {
                    //we check whether the node is orphaned, and if it is we don't build any edges for it
                    if !node.is_orphan(&self.nodetemplates) {
                        //we get the node's id from the id map
                        //we iterate over the nodeids, ensure that there aren't any duplicates, and push each pair of nodeids into edges
                        for rhs in &nodestringids {
                            let nodeid = node_id_map.get(&node.id).unwrap();
                            let rhsid = node_id_map.get(rhs).unwrap();
                            if !system
                                .nodes
                                .iter()
                                .find(|n| n.id == *rhs)
                                .unwrap()
                                .is_orphan(&self.nodetemplates)
                                && !(self.edges.iter().any(|(nodeid1, nodeid2, _)| {
                                    (nodeid1 == &node.id && nodeid2 == rhs) || (nodeid2 == &node.id && nodeid1 == rhs)
                                }))
                            {
                                let flavor = if self.edges.iter().any(|(nodeid1, nodeid2, _)| {
                                    nodeid1 == &node.id
                                        || nodeid1 == rhs
                                        || nodeid2 == &node.id
                                        || nodeid2 == rhs
                                }) {
                                    edgeflavor_id_map.get(&self.semi_internal_edgeflavor).expect("Specified semi_internal_edgeflavor is not a valid edgeflavor id!").clone()
                                } else {
                                    edgeflavor_id_map.get(&self.pure_internal_edgeflavor).expect("Specified pure_internal_edgeflavor is not a valid edgeflavor id!").clone()
                                };
                                assert_ne!(nodeid, rhsid, "Same node ID appears twice.");
                                edges.insert(
                                    (nodeid.min(rhsid).clone(), nodeid.max(rhsid).clone()),
                                    flavor,
                                );
                            }
                        }
                    }
                    nodestringids.push(node.id.clone());
                }
                (system.id.clone(), Arc::new(system.hydrate(i, &node_id_map)))
            })
            .collect();

        let neighbors: HashMap<Arc<internal::Node>, Vec<Arc<internal::Node>>> =
            edges.iter().fold(HashMap::new(), |mut acc, ((a, b), _)| {
                acc.entry(a.clone())
                    .or_insert_with(Vec::new)
                    .push(b.clone());
                acc.entry(b.clone())
                    .or_insert_with(Vec::new)
                    .push(a.clone());
                acc
            });

        let squadronflavor_id_map: HashMap<String, Arc<internal::SquadronFlavor>> = self
            .squadronflavors
            .drain(0..)
            .enumerate()
            .map(|(i, squadronflavor)| {
                (
                    squadronflavor.id.clone(),
                    Arc::new(squadronflavor.hydrate(i)),
                )
            })
            .collect();

        let squadronclasses: HashMap<String, Arc<internal::SquadronClass>> = self
            .squadronclasses
            .iter()
            .enumerate()
            .map(|(i, squadronclass)| {
                (
                    squadronclass.id.clone(),
                    Arc::new(squadronclass.hydrate(
                        i,
                        &shipclass_id_map,
                        &squadronclass_id_map,
                        &squadronflavor_id_map,
                        &factions,
                        &self.shipclasses,
                        &self.squadronclasses,
                    )),
                )
            })
            .collect();

        internal::Root {
            config: config,
            nodeflavors: nodeflavor_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            nodes: node_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            systems: system_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            edgeflavors: edgeflavor_id_map
                .values()
                .sorted_by_key(|x| x.id)
                .cloned()
                .collect(),
            edges,
            neighbors,
            factions: factions.values().cloned().sorted_by_key(|x| x.id).collect(),
            wars,
            resources: resource_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            hangarclasses: hangarclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            hangarcounter: Arc::new(AtomicU64::new(0)),
            engineclasses: engineclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            repairerclasses: repairerclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            strategicweaponclasses: strategicweaponclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            factoryclasses: factoryclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            shipyardclasses: shipyardclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            subsystemclasses: subsystemclass_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            shipais: shipai_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            shipflavors: shipflavor_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            squadronflavors: squadronflavor_id_map
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            shipclasses: shipclasses.clone(),
            squadronclasses: squadronclasses
                .values()
                .cloned()
                .sorted_by_key(|x| x.id)
                .collect(),
            ships: RwLock::new(Vec::new()),
            squadrons: RwLock::new(Vec::new()),
            unitcounter: Arc::new(AtomicU64::new(0)),
            engagements: RwLock::new(Vec::new()),
            global_salience: internal::GlobalSalience {
                faction_salience: RwLock::new(
                    factions
                        .iter()
                        .map(|_| {
                            factions
                                .iter()
                                .map(|_| node_id_map.iter().map(|_| [0.0; 2]).collect())
                                .collect()
                        })
                        .collect(),
                ),
                resource_salience: RwLock::new(
                    factions
                        .iter()
                        .map(|_| {
                            resource_id_map
                                .iter()
                                .map(|_| node_id_map.iter().map(|_| [0.0; 2]).collect())
                                .collect()
                        })
                        .collect(),
                ),
                unitclass_salience: RwLock::new(
                    factions
                        .iter()
                        .map(|_| {
                            shipclasses
                                .iter()
                                .map(|_| node_id_map.iter().map(|_| [0.0; 2]).collect())
                                .collect()
                        })
                        .collect(),
                ),
                strategic_weapon_effect_map: RwLock::new(
                    factions
                        .iter()
                        .map(|_| node_id_map.iter().map(|_| [(0, 0.0); 3]).collect())
                        .collect(),
                ),
            },
            turn: Arc::new(AtomicU64::new(0)),
        }
    }
}
