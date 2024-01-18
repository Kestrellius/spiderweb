//this is the section of the program that manages the json files defined by the modder
use crate::internal;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::iter;
use std::sync::atomic::{self, AtomicU64, AtomicUsize};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    saliencescalars: SalienceScalars,
    entityscalars: EntityScalars,
    battlescalars: BattleScalars,
}

impl Config {
    fn hydrate(self) -> internal::Config {
        internal::Config {
            saliencescalars: self.saliencescalars.hydrate(),
            entityscalars: self.entityscalars.hydrate(),
            battlescalars: self.battlescalars.hydrate(),
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntityScalars {
    defect_escape_scalar: Option<f32>,
    victor_morale_scalar: Option<f32>,
    victis_morale_scalar: Option<f32>,
}

impl EntityScalars {
    fn hydrate(self) -> internal::EntityScalars {
        internal::EntityScalars {
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
            duration_damage_scalar: self.duration_damage_scalar.unwrap_or(1.0),
        }
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct NodeFlavor {
    id: String,
    visiblename: String,
    description: String,
}

impl NodeFlavor {
    fn hydrate(self, index: usize) -> internal::NodeFlavor {
        internal::NodeFlavor {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct NodeTemplate {
    id: String,
    visiblename: String,
    description: String,
    visibility: bool,
    flavor: Vec<(String, u64)>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    factorylist: HashMap<String, (f32, u64, u64)>, //a list of the factories this node has, in the form of FactoryClass IDs
    shipyardlist: HashMap<String, (f32, u64, u64)>,
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
    visiblename: Option<String>, //location name as shown to player
    position: Option<[i64; 3]>, //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    description: Option<String>,
    visibility: Option<bool>,
    flavor: Option<String>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    factorylist: Option<Vec<String>>, //a list of the factories this node has, in the form of FactoryClass IDs
    shipyardlist: Option<Vec<String>>,
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
        nodeflavoridmap: &HashMap<String, Arc<internal::NodeFlavor>>,
        factions: &HashMap<String, Arc<internal::Faction>>,
        factoryclassidmap: &HashMap<String, Arc<internal::FactoryClass>>,
        shipyardclassidmap: &HashMap<String, Arc<internal::ShipyardClass>>,
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
            visiblename: self
                .visiblename
                .clone()
                .unwrap_or(template.visiblename)
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
                flavor: nodeflavoridmap
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
                units: Vec::new(),
                factoryinstancelist: match &self.factorylist {
                    Some(list) => list
                        .iter()
                        .map(|stringid| {
                            internal::FactoryClass::instantiate(
                                factoryclassidmap
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
                        .factorylist
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
                                        factoryclassidmap
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
                shipyardinstancelist: match &self.shipyardlist {
                    Some(list) => list
                        .iter()
                        .map(|stringid| {
                            internal::ShipyardClass::instantiate(
                                shipyardclassidmap
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
                        .shipyardlist
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
                                        shipyardclassidmap
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
                    .unwrap_or(template.efficiency.unwrap_or(faction.efficiencydefault)),
                balance_stockpiles: self
                    .balance_stockpiles
                    .unwrap_or(template.balance_stockpiles.unwrap_or(true)),
                balance_hangars: self
                    .balance_hangars
                    .unwrap_or(template.balance_hangars.unwrap_or(true)),
                check_for_battles: self
                    .check_for_battles
                    .unwrap_or(template.check_for_battles.unwrap_or(true)),
                stockpiles_balanced: false,
                hangars_balanced: false,
            }),
        };
        node
    }
}

#[derive(Serialize, Deserialize)]
struct System {
    id: String,
    visiblename: String,
    description: String,
    visibility: Option<bool>,
    nodes: Vec<Node>,
}

impl System {
    fn hydrate(
        self,
        index: usize,
        nodeidmap: &HashMap<String, Arc<internal::Node>>,
    ) -> internal::System {
        let internalsystem = internal::System {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            nodes: self
                .nodes
                .iter()
                .map(|node| nodeidmap.get(&node.id).unwrap().clone())
                .collect(),
        };
        internalsystem
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct EdgeFlavor {
    id: String,
    visiblename: String,
    description: String,
    propagates: bool,
}

impl EdgeFlavor {
    fn hydrate(self, index: usize) -> internal::EdgeFlavor {
        internal::EdgeFlavor {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            propagates: self.propagates,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Faction {
    id: String,
    visiblename: String, //faction name as shown to player
    description: String,
    visibility: Option<bool>,
    propagates: Option<bool>,
    efficiencydefault: f32, //starting value for production facility efficiency
    efficiencytarget: f32, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    efficiencydelta: f32,  //rate at which efficiency changes
    battlescalar: f32,
    relations: HashMap<String, f32>,
}

impl Faction {
    fn hydrate(
        self,
        index: usize,
        factionidmap: &HashMap<String, internal::FactionID>,
    ) -> internal::Faction {
        let faction = internal::Faction {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            efficiencydefault: self.efficiencydefault,
            efficiencytarget: self.efficiencytarget,
            efficiencydelta: self.efficiencydelta,
            battlescalar: self.battlescalar,
            relations: factionidmap
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
    visiblename: String,
    description: String,
    visibility: Option<bool>,
    propagates: Option<bool>,
    unit_vol: u64, //how much space a one unit of this resource takes up when transported by a cargo ship
    valuemult: u64, //how valuable the AI considers one unit of this resource to be
}

impl Resource {
    fn hydrate(self, index: usize) -> internal::Resource {
        let resource = internal::Resource {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            unit_vol: self.unit_vol,
            valuemult: self.valuemult,
        };
        resource
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnipotentStockpile {
    visibility: Option<bool>,
    resourcetype: String,
    contents: u64,
    rate: Option<u64>,
    target: u64,
    capacity: u64,
    propagates: Option<bool>,
}

impl UnipotentStockpile {
    fn hydrate(
        self,
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::UnipotentStockpile {
        let stockpile = internal::UnipotentStockpile {
            visibility: self.visibility.unwrap_or(true),
            resourcetype: resourceidmap
                .get(&self.resourcetype)
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
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::PluripotentStockpile {
        let stockpile = internal::PluripotentStockpile {
            visibility: self.visibility.unwrap_or(true),
            contents: self
                .contents
                .drain()
                .map(|(id, num)| {
                    (
                        resourceidmap
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
                    .map(|id| resourceidmap.get(id).expect("Resource is invalid!").clone())
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
    visiblename: String,
    description: String,
    visibility: Option<bool>,
    capacity: u64,               //total volume the hangar can hold
    target: u64, //volume the hangar wants to hold; this is usually either equal to capacity (for carriers) or zero (for shipyard outputs)
    allowed: Vec<String>, //which shipclasses this hangar can hold
    ideal: HashMap<String, u64>, //how many of each ship type the hangar wants
    non_ideal_supply_scalar: Option<f32>,
    launch_volume: u64, //how much volume the hangar can launch at one time in battle
    launch_interval: u64, //time between launches in battle
    propagates: Option<bool>, //whether or not hangar generates saliences
}

impl HangarClass {
    fn hydrate(
        mut self,
        index: usize,
        shipclassidmap: &HashMap<String, internal::ShipClassID>,
        squadronclassidmap: &HashMap<String, internal::SquadronClassID>,
    ) -> internal::HangarClass {
        let hangarclass = internal::HangarClass {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            capacity: self.capacity,
            target: self.target,
            allowed: self
                .allowed
                .drain(0..)
                .map(|x| {
                    if let Some(shipclassid) = shipclassidmap.get(&x) {
                        internal::UnitClassID::ShipClass(*shipclassid)
                    } else {
                        internal::UnitClassID::SquadronClass(*squadronclassidmap.get(&x).unwrap())
                    }
                })
                .collect(),
            ideal: self
                .ideal
                .drain()
                .map(|(k, v)| {
                    (
                        {
                            if let Some(shipclassid) = shipclassidmap.get(&k) {
                                internal::UnitClassID::ShipClass(*shipclassid)
                            } else {
                                internal::UnitClassID::SquadronClass(
                                    *squadronclassidmap.get(&k).unwrap(),
                                )
                            }
                        },
                        v,
                    )
                })
                .collect(),
            non_ideal_supply_scalar: self.non_ideal_supply_scalar.unwrap_or(0.5),
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
    visiblename: String,
    description: String,
    visibility: Option<bool>,
    basehealth: Option<u64>,
    toughnessscalar: f32,
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
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
        nodeflavoridmap: &HashMap<String, Arc<internal::NodeFlavor>>,
        edgeflavoridmap: &HashMap<String, Arc<internal::EdgeFlavor>>,
    ) -> internal::EngineClass {
        internal::EngineClass {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            basehealth: self.basehealth,
            toughnessscalar: self.toughnessscalar,
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resourceidmap))
                .collect(),
            forbidden_nodeflavors: match self.forbidden_nodeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| nodeflavoridmap.get(&s).unwrap().clone())
                    .collect(),
                None => Vec::new(),
            },
            forbidden_edgeflavors: match self.forbidden_edgeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| edgeflavoridmap.get(&s).unwrap().clone())
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
    visiblename: String,
    description: String,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>,
    repair_points: i64,
    repair_factor: f32,
    engine_repair_points: i64,
    engine_repair_factor: f32,
    per_engagement: Option<bool>, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerClass {
    fn hydrate(
        mut self,
        index: usize,
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::RepairerClass {
        let repairerclass = internal::RepairerClass {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resourceidmap))
                .collect(),
            repair_points: self.repair_points,
            repair_factor: self.repair_factor,
            engine_repair_points: self.engine_repair_points,
            engine_repair_factor: self.engine_repair_factor,
            per_engagement: self.per_engagement.unwrap_or(false),
        };
        repairerclass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FactoryClass {
    id: String,
    visiblename: String,
    description: String,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    fn hydrate(
        mut self,
        index: usize,
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
    ) -> internal::FactoryClass {
        let factoryclass = internal::FactoryClass {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resourceidmap))
                .collect(),
            outputs: self
                .outputs
                .drain(0..)
                .map(|x| x.hydrate(resourceidmap))
                .collect(),
        };
        factoryclass
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShipyardClass {
    id: String,
    visiblename: Option<String>,
    description: Option<String>,
    visibility: Option<bool>,
    inputs: Vec<UnipotentStockpile>,
    outputs: HashMap<String, u64>,
    constructrate: u64,
    efficiency: f32,
}

impl ShipyardClass {
    fn hydrate(
        mut self,
        index: usize,
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::ShipClassID>,
    ) -> internal::ShipyardClass {
        let shipyardclass = internal::ShipyardClass {
            id: index,
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resourceidmap))
                .collect(),
            outputs: self
                .outputs
                .drain()
                .map(|(k, v)| (*shipclassidmap.get(&k).unwrap(), v))
                .collect(),
            constructrate: self.constructrate,
            efficiency: self.efficiency,
        };
        shipyardclass
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShipAI {
    id: String,
    ship_attract_specific: f32, //a multiplier for supply gradients corresponding to the specific class of a ship using this AI
    ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship supply gradients
    ship_cargo_attract: HashMap<String, f32>, //a list of ship classes whose supply gradients this AI will follow (so as to carry e.g. fighters that can't travel on their own), and individual strength multipliers
    resource_attract: HashMap<String, f32>, //a list of resources whose supply gradients this AI will follow, and individual strength multipliers
}

impl ShipAI {
    fn hydrate(
        self,
        index: usize,
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::ShipClassID>,
    ) -> internal::ShipAI {
        let shipai = internal::ShipAI {
            id: index,
            ship_attract_specific: self.ship_attract_specific,
            ship_attract_generic: self.ship_attract_generic,
            ship_cargo_attract: self
                .ship_cargo_attract
                .iter()
                .map(|(stringid, v)| {
                    (
                        internal::UnitClassID::ShipClass(*shipclassidmap.get(stringid).unwrap()),
                        *v,
                    )
                })
                .collect(),
            resource_attract: self
                .resource_attract
                .iter()
                .map(|(stringid, v)| (resourceidmap.get(stringid).unwrap().clone(), *v))
                .collect(),
        };
        shipai
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShipFlavor {
    pub id: String,
    pub visiblename: String,
    pub description: String,
}

impl ShipFlavor {
    fn hydrate(self, index: usize) -> internal::ShipFlavor {
        internal::ShipFlavor {
            id: index,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ShipClass {
    id: String,
    visiblename: String,
    description: String,
    shipflavor: String,
    basehull: u64,     //how many hull hitpoints this ship has by default
    basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    visibility: Option<bool>,
    propagates: Option<bool>,
    hangarvol: u64, //how much hangar space this ship takes up when carried by a host
    stockpiles: Option<Vec<PluripotentStockpile>>,
    defaultweapons: Option<HashMap<String, u64>>, //a strikecraft's default weapons, which it always has with it
    hangars: Option<Vec<String>>,
    engines: Option<Vec<String>>,
    repairers: Option<Vec<String>>,
    factoryclasslist: Option<Vec<String>>,
    shipyardclasslist: Option<Vec<String>>,
    aiclass: String,
    navthreshold: Option<f32>,
    processordemandnavscalar: Option<f32>,
    deploys_self: Option<bool>,
    deploys_daughters: Option<Option<u64>>,
    defectchance: Option<HashMap<String, (f32, f32)>>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    toughnessscalar: Option<f32>,
    battleescapescalar: Option<f32>,
    defectescapescalar: Option<f32>,
    interdictionscalar: Option<f32>,
}

impl ShipClass {
    fn hydrate(
        &self,
        shipclassidmap: &HashMap<String, internal::ShipClassID>,
        shipflavoridmap: &HashMap<String, Arc<internal::ShipFlavor>>,
        resourceidmap: &HashMap<String, Arc<internal::Resource>>,
        hangarclassidmap: &HashMap<String, Arc<internal::HangarClass>>,
        engineclassidmap: &HashMap<String, Arc<internal::EngineClass>>,
        repairerclassidmap: &HashMap<String, Arc<internal::RepairerClass>>,
        factoryclassidmap: &HashMap<String, Arc<internal::FactoryClass>>,
        shipyardclassidmap: &HashMap<String, Arc<internal::ShipyardClass>>,
        shipaiidmap: &HashMap<String, Arc<internal::ShipAI>>,
        factions: &HashMap<String, Arc<internal::Faction>>,
    ) -> internal::ShipClass {
        let shipclass = internal::ShipClass {
            id: shipclassidmap.get(&self.id).unwrap().index,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            shipflavor: shipflavoridmap.get(&self.shipflavor).unwrap().clone(),
            basehull: self.basehull,
            basestrength: self.basestrength,
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            defaultweapons: self.defaultweapons.clone().map(|map| {
                map.iter()
                    .map(|(id, n)| {
                        (
                            resourceidmap
                                .get(id)
                                .unwrap_or_else(|| panic!("{} is not found!", id))
                                .clone(),
                            *n,
                        )
                    })
                    .collect()
            }),
            hangarvol: self.hangarvol,
            stockpiles: self
                .stockpiles
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|stockpile| stockpile.clone().hydrate(&resourceidmap))
                .collect(),
            hangars: self
                .hangars
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    hangarclassidmap
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
                    engineclassidmap
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
                    repairerclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            factoryclasslist: self
                .factoryclasslist
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    factoryclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            shipyardclasslist: self
                .shipyardclasslist
                .clone()
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    shipyardclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                        .clone()
                })
                .collect(),
            aiclass: shipaiidmap.get(&self.aiclass).unwrap().clone(),
            processordemandnavscalar: self.processordemandnavscalar.unwrap_or(10.0),
            navthreshold: self.navthreshold.unwrap_or(1.0),
            deploys_self: self.deploys_self.unwrap_or(true),
            deploys_daughters: self.deploys_daughters.unwrap_or(None),
            defectchance: self
                .defectchance
                .clone()
                .unwrap_or(HashMap::new())
                .iter()
                .map(|(k, v)| (factions.get(k).unwrap().clone(), *v))
                .collect(),
            toughnessscalar: self.toughnessscalar.unwrap_or(1.0),
            battleescapescalar: self.battleescapescalar.unwrap_or(1.0),
            defectescapescalar: self.defectescapescalar.unwrap_or(1.0),
            interdictionscalar: self.interdictionscalar.unwrap_or(1.0),
        };
        shipclass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquadronFlavor {
    pub id: String,
    pub visiblename: String,
    pub description: String,
}

impl SquadronFlavor {
    fn hydrate(self, index: usize) -> internal::SquadronFlavor {
        internal::SquadronFlavor {
            id: index,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SquadronClass {
    id: String,
    visiblename: String,
    description: String,
    squadronflavor: String,
    visibility: Option<bool>,
    propagates: Option<bool>,
    strengthmod: (f32, u64),
    squadronconfig: HashMap<String, u64>,
    non_ideal_supply_scalar: Option<f32>,
    navthreshold: Option<f32>,
    navquorum: f32,
    disbandthreshold: f32,
    deploys_self: Option<bool>,
    deploys_daughters: Option<Option<u64>>,
    defectchance: Option<HashMap<String, (f32, f32)>>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    defectescapescalar: Option<f32>,
}

impl SquadronClass {
    fn get_target(
        &self,
        shipclasses: &Vec<ShipClass>,
        squadronclasses: &Vec<SquadronClass>,
    ) -> u64 {
        self.squadronconfig
            .iter()
            .map(|(stringid, _)| {
                if let Some(shipclass) = shipclasses.iter().find(|sc| sc.id == *stringid) {
                    shipclass.hangarvol
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
        shipclassidmap: &HashMap<String, internal::ShipClassID>,
        squadronclassidmap: &HashMap<String, internal::SquadronClassID>,
        squadronflavoridmap: &HashMap<String, Arc<internal::SquadronFlavor>>,
        factions: &HashMap<String, Arc<internal::Faction>>,
        shipclasses: &Vec<ShipClass>,
        squadronclasses: &Vec<SquadronClass>,
    ) -> internal::SquadronClass {
        let squadronclass = internal::SquadronClass {
            id: index,
            visiblename: self.visiblename.clone(),
            description: self.description.clone(),
            squadronflavor: squadronflavoridmap
                .get(&self.squadronflavor)
                .unwrap()
                .clone(),
            visibility: self.visibility.unwrap_or(true),
            propagates: self.propagates.unwrap_or(true),
            strengthmod: self.strengthmod.clone(),
            squadronconfig: self
                .squadronconfig
                .iter()
                .map(|(stringid, n)| {
                    (
                        {
                            if let Some(shipclassid) = shipclassidmap.get(&stringid.clone()) {
                                internal::UnitClassID::ShipClass(*shipclassid)
                            } else {
                                internal::UnitClassID::SquadronClass(
                                    *squadronclassidmap.get(&stringid.clone()).unwrap(),
                                )
                            }
                        },
                        *n,
                    )
                })
                .collect(),
            non_ideal_supply_scalar: self.non_ideal_supply_scalar.unwrap_or(0.5),
            target: self.get_target(shipclasses, squadronclasses),
            navthreshold: self.navthreshold.unwrap_or(1.0),
            navquorum: self.navquorum,
            disbandthreshold: self.disbandthreshold,
            deploys_self: self.deploys_self.unwrap_or(true),
            deploys_daughters: self.deploys_daughters.unwrap_or(None),
            defectchance: self
                .defectchance
                .clone()
                .unwrap_or(HashMap::new())
                .iter()
                .map(|(stringid, n)| (factions.get(stringid).unwrap().clone(), *n))
                .collect(),
            defectescapescalar: self.defectescapescalar.unwrap_or(1.0),
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
    factoryclasses: Vec<FactoryClass>,
    shipyardclasses: Vec<ShipyardClass>,
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

        let nodeflavoridmap: HashMap<String, Arc<internal::NodeFlavor>> = self
            .nodeflavors
            .drain(0..)
            .enumerate()
            .map(|(i, nodeflavor)| (nodeflavor.id.clone(), Arc::new(nodeflavor.hydrate(i))))
            .collect();

        let edgeflavoridmap: HashMap<String, Arc<internal::EdgeFlavor>> = self
            .edgeflavors
            .drain(0..)
            .enumerate()
            .map(|(i, edgeflavor)| (edgeflavor.id.clone(), Arc::new(edgeflavor.hydrate(i))))
            .collect();

        let factionidmap: HashMap<String, internal::FactionID> = self
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
                //we make sure the enumeration index we have matches the faction's entry in the idmap
                assert_eq!(i, factionidmap.get(&faction.id).unwrap().index);
                (
                    faction.id.clone(),
                    Arc::new(faction.hydrate(i, &factionidmap)),
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
        let resourceidmap: HashMap<String, Arc<internal::Resource>> = self
            .resources
            .drain(0..)
            .enumerate()
            .map(|(i, resource)| (resource.id.clone(), Arc::new(resource.hydrate(i))))
            .collect();

        let engineclassidmap: HashMap<String, Arc<internal::EngineClass>> = self
            .engineclasses
            .drain(0..)
            .enumerate()
            .map(|(i, engineclass)| {
                (
                    engineclass.id.clone(),
                    Arc::new(engineclass.hydrate(
                        i,
                        &resourceidmap,
                        &nodeflavoridmap,
                        &edgeflavoridmap,
                    )),
                )
            })
            .collect();

        let repairerclassidmap: HashMap<String, Arc<internal::RepairerClass>> = self
            .repairerclasses
            .drain(0..)
            .enumerate()
            .map(|(i, repairerclass)| {
                (
                    repairerclass.id.clone(),
                    Arc::new(repairerclass.hydrate(i, &resourceidmap)),
                )
            })
            .collect();

        let factoryclassidmap: HashMap<String, Arc<internal::FactoryClass>> = self
            .factoryclasses
            .drain(0..)
            .enumerate()
            .map(|(i, factoryclass)| {
                (
                    factoryclass.id.clone(),
                    Arc::new(factoryclass.hydrate(i, &resourceidmap)),
                )
            })
            .collect();

        //this is a dummy ship class, which is here so that salience processes that require a shipclass to be specified can be parsed correctly
        let generic_demand_ship = ShipClass {
            id: "generic_demand_ship".to_string(),
            visiblename: "Generic Demand Ship".to_string(),
            description: "".to_string(),
            shipflavor: "default".to_string(),
            basehull: 1,     //how many hull hitpoints this ship has by default
            basestrength: 0, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
            visibility: Some(false),
            propagates: Some(false),
            hangarvol: 0, //how much hangar space this ship takes up when carried by a host
            stockpiles: None,
            defaultweapons: None, //a strikecraft's default weapons, which it always has with it
            hangars: None,
            engines: None,
            repairers: None,
            factoryclasslist: None,
            shipyardclasslist: None,
            aiclass: "default".to_string(), //aiclass
            navthreshold: None,
            processordemandnavscalar: None,
            deploys_self: None,
            deploys_daughters: None,
            defectchance: None,
            toughnessscalar: None,
            battleescapescalar: None,
            defectescapescalar: None,
            interdictionscalar: None,
        };

        //here we create the shipclassidmap, put the dummy ship class inside it, and then insert all the actual ship classes
        let shipclassidmap: HashMap<String, internal::ShipClassID> =
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

        let squadronclassidmap: HashMap<String, internal::SquadronClassID> = self
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

        let hangarclassidmap: HashMap<String, Arc<internal::HangarClass>> = self
            .hangarclasses
            .drain(0..)
            .enumerate()
            .map(|(i, hangarclass)| {
                (
                    hangarclass.id.clone(),
                    Arc::new(hangarclass.hydrate(i, &shipclassidmap, &squadronclassidmap)),
                )
            })
            .collect();

        let shipyardclassidmap: HashMap<String, Arc<internal::ShipyardClass>> = self
            .shipyardclasses
            .drain(0..)
            .enumerate()
            .map(|(i, shipyardclass)| {
                (
                    shipyardclass.id.clone(),
                    Arc::new(shipyardclass.hydrate(i, &resourceidmap, &shipclassidmap)),
                )
            })
            .collect();

        let shipaiidmap: HashMap<String, Arc<internal::ShipAI>> = self
            .shipais
            .drain(0..)
            .enumerate()
            .map(|(i, shipai)| {
                (
                    shipai.id.clone(),
                    Arc::new(shipai.hydrate(i, &resourceidmap, &shipclassidmap)),
                )
            })
            .collect();

        let shipflavoridmap: HashMap<String, Arc<internal::ShipFlavor>> = self
            .shipflavors
            .drain(0..)
            .enumerate()
            .map(|(i, shipflavor)| (shipflavor.id.clone(), Arc::new(shipflavor.hydrate(i))))
            .collect();

        //we hydrate shipclasses, starting with the generic demand ship
        let shipclasses: Vec<Arc<internal::ShipClass>> = iter::once(&generic_demand_ship)
            .chain(self.shipclasses.iter())
            .map(|shipclass| {
                Arc::new(shipclass.hydrate(
                    &shipclassidmap,
                    &shipflavoridmap,
                    &resourceidmap,
                    &hangarclassidmap,
                    &engineclassidmap,
                    &repairerclassidmap,
                    &factoryclassidmap,
                    &shipyardclassidmap,
                    &shipaiidmap,
                    &factions,
                ))
            })
            .collect();

        let mut rng = rand_hc::Hc128Rng::seed_from_u64(1138);

        //here we iterate over the json systems to create a map between nodes' json string-ids and internal ids
        let nodeidmap: HashMap<String, Arc<internal::Node>> = self
            .systems
            .iter()
            .flat_map(|system| system.nodes.iter())
            .enumerate()
            .map(|(i, node)| {
                let nodehydration = node.hydrate(
                    i,
                    &mut rng,
                    &self.nodetemplates,
                    &nodeflavoridmap,
                    &factions,
                    &factoryclassidmap,
                    &shipyardclassidmap,
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
                let aid = nodeidmap.get(a).unwrap().clone();
                let bid = nodeidmap.get(b).unwrap().clone();
                assert_ne!(aid, bid);
                (
                    (aid.clone().min(bid.clone()), bid.max(aid)),
                    edgeflavoridmap.get(f).unwrap().clone(),
                )
            })
            .collect();

        let systemidmap: HashMap<String, Arc<internal::System>> = self
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
                            let nodeid = nodeidmap.get(&node.id).unwrap();
                            let rhsid = nodeidmap.get(rhs).unwrap();
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
                                    edgeflavoridmap.get(&self.semi_internal_edgeflavor).expect("Specified semi_internal_edgeflavor is not a valid edgeflavor id!").clone()
                                } else {
                                    edgeflavoridmap.get(&self.pure_internal_edgeflavor).expect("Specified pure_internal_edgeflavor is not a valid edgeflavor id!").clone()
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
                (system.id.clone(), Arc::new(system.hydrate(i, &nodeidmap)))
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

        let squadronflavoridmap: HashMap<String, Arc<internal::SquadronFlavor>> = self
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
                        &shipclassidmap,
                        &squadronclassidmap,
                        &squadronflavoridmap,
                        &factions,
                        &self.shipclasses,
                        &self.squadronclasses,
                    )),
                )
            })
            .collect();

        internal::Root {
            config: config,
            nodeflavors: nodeflavoridmap.values().cloned().collect(),
            nodes: nodeidmap.values().cloned().collect(),
            systems: systemidmap.values().cloned().collect(),
            edgeflavors: edgeflavoridmap.values().cloned().collect(),
            edges,
            neighbors,
            factions: factions.values().cloned().collect(),
            wars,
            resources: resourceidmap.values().cloned().collect(),
            hangarclasses: hangarclassidmap.values().cloned().collect(),
            hangarinstancecounter: Arc::new(AtomicU64::new(0)),
            engineclasses: engineclassidmap.values().cloned().collect(),
            repairerclasses: repairerclassidmap.values().cloned().collect(),
            factoryclasses: factoryclassidmap.values().cloned().collect(),
            shipyardclasses: shipyardclassidmap.values().cloned().collect(),
            shipais: shipaiidmap.values().cloned().collect(),
            shipflavors: shipflavoridmap.values().cloned().collect(),
            squadronflavors: squadronflavoridmap.values().cloned().collect(),
            shipclasses: shipclasses.clone(),
            squadronclasses: squadronclasses.values().cloned().collect(),
            shipinstances: RwLock::new(Vec::new()),
            squadroninstances: RwLock::new(Vec::new()),
            unitcounter: Arc::new(AtomicU64::new(0)),
            engagements: RwLock::new(Vec::new()),
            globalsalience: internal::GlobalSalience {
                factionsalience: RwLock::new(
                    factions
                        .iter()
                        .map(|_| {
                            factions
                                .iter()
                                .map(|_| nodeidmap.iter().map(|_| [0.0; 2]).collect())
                                .collect()
                        })
                        .collect(),
                ),
                resourcesalience: RwLock::new(
                    factions
                        .iter()
                        .map(|_| {
                            resourceidmap
                                .iter()
                                .map(|_| nodeidmap.iter().map(|_| [0.0; 2]).collect())
                                .collect()
                        })
                        .collect(),
                ),
                unitclasssalience: RwLock::new(
                    factions
                        .iter()
                        .map(|_| {
                            shipclasses
                                .iter()
                                .map(|_| nodeidmap.iter().map(|_| [0.0; 2]).collect())
                                .collect()
                        })
                        .collect(),
                ),
            },
            turn: Arc::new(AtomicU64::new(0)),
        }
    }
}
