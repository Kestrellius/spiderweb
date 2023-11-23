//this is the section of the program that manages the json files defined by the modder
use crate::internal;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer_pretty};
use std::collections::{HashMap, HashSet};
use std::iter;

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct NodeFlavor {
    id: String,
    visiblename: String,
    description: String,
}

impl NodeFlavor {
    fn hydrate(self) -> internal::NodeFlavor {
        internal::NodeFlavor {
            visiblename: self.visiblename,
            description: self.description,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Node {
    id: String,
    visiblename: String,        //location name as shown to player
    position: Option<[i64; 3]>, //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    description: String,
    flavor: String, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    factorylist: Vec<String>, //a list of the factories this node has, in the form of FactoryClass IDs
    shipyardlist: Vec<String>,
    environment: String, //name of the FRED environment to use for missions set in this node
    bitmap: Option<(String, f32)>,
    orphan: Option<bool>, //an orphaned node does not get all-to-all edges automatically built with the other nodes in its system
    allegiance: String,   //faction that currently holds the node
    efficiency: Option<f64>, //efficiency of any production facilities in this node; changes over time based on faction ownership
}

impl Node {
    fn hydrate(
        self,
        nodeflavoridmap: &HashMap<String, internal::Key<internal::NodeFlavor>>,
        factionidmap: &HashMap<String, internal::Key<internal::Faction>>,
        factoryclasses: &internal::Table<internal::FactoryClass>,
        factoryclassidmap: &HashMap<String, internal::Key<internal::FactoryClass>>,
        shipyardclasses: &internal::Table<internal::ShipyardClass>,
        shipyardclassidmap: &HashMap<String, internal::Key<internal::ShipyardClass>>,
    ) -> (String, internal::Node) {
        let node = internal::Node {
            visiblename: self.visiblename,
            system: internal::Key::<internal::System>::new_from_index(0),
            position: self.position.unwrap_or([0, 0, 0]),
            description: self.description,
            flavor: *nodeflavoridmap
                .get(&self.flavor)
                .expect("Node flavor field is not correctly defined!"),
            factoryinstancelist: self
                .factorylist
                .iter()
                .map(|stringid| {
                    let classid = factoryclassidmap.get(stringid).unwrap();
                    factoryclasses.get(*classid).unwrap().instantiate()
                })
                .collect(),
            shipyardinstancelist: self
                .shipyardlist
                .iter()
                .map(|stringid| {
                    let classid = shipyardclassidmap
                        .get(stringid)
                        .expect(&format!("Shipyard '{}' does not exist.", stringid));
                    shipyardclasses.get(*classid).unwrap().instantiate()
                })
                .collect(),
            environment: self.environment,
            bitmap: self.bitmap,
            allegiance: *factionidmap
                .get(&self.allegiance)
                .expect("Allegiance field is not correctly defined!"),
            efficiency: self.efficiency.unwrap_or(1.0),
            threat: factionidmap.values().map(|&id| (id, 0_f32)).collect(),
        };
        (self.id, node)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct System {
    id: String,
    visiblename: String,
    description: String,
    nodes: Vec<Node>,
}

impl System {
    fn hydrate(
        self,
        nodeidmap: &HashMap<String, internal::Key<internal::Node>>,
    ) -> (String, internal::System, Vec<Node>) {
        let internalsystem = internal::System {
            visiblename: self.visiblename,
            description: self.description,
            nodes: self
                .nodes
                .iter()
                .map(|node| *nodeidmap.get(&node.id).unwrap())
                .collect(),
        };
        (self.id, internalsystem, self.nodes)
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct EdgeFlavor {
    id: String,
    visiblename: String,
    description: String,
}

impl EdgeFlavor {
    fn hydrate(self) -> internal::EdgeFlavor {
        internal::EdgeFlavor {
            visiblename: self.visiblename,
            description: self.description,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Faction {
    id: String,
    visiblename: String, //faction name as shown to player
    description: String,
    efficiencydefault: f64, //starting value for production facility efficiency
    efficiencytarget: f64, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    efficiencydelta: f64,  //rate at which efficiency changes
    battlescalar: f32,
    relations: HashMap<String, f32>,
}

impl Faction {
    fn hydrate(
        self,
        factionidmap: &HashMap<String, internal::Key<internal::Faction>>,
    ) -> internal::Faction {
        let faction = internal::Faction {
            visiblename: self.visiblename,
            description: self.description,
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
    cargovol: u64, //how much space a one unit of this resource takes up when transported by a cargo ship
    valuemult: u64, //how valuable the AI considers one unit of this resource to be
}

impl Resource {
    fn hydrate(self) -> (String, internal::Resource) {
        let resource = internal::Resource {
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            cargovol: self.cargovol,
            valuemult: self.valuemult,
        };
        (self.id, resource)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnipotentResourceStockpile {
    visibility: Option<bool>,
    resourcetype: String,
    contents: u64,
    rate: Option<u64>,
    target: u64,
    capacity: u64,
    propagate: Option<bool>,
}

impl UnipotentResourceStockpile {
    fn hydrate(
        self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
    ) -> internal::UnipotentResourceStockpile {
        let stockpile = internal::UnipotentResourceStockpile {
            visibility: self.visibility.unwrap_or(true),
            resourcetype: *resourceidmap
                .get(&self.resourcetype)
                .expect("Resource is invalid!"),
            contents: self.contents,
            rate: self.rate.unwrap_or(0),
            target: self.target,
            capacity: self.capacity,
            propagate: self.propagate.unwrap_or(true),
        };
        stockpile
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluripotentStockpile {
    visibility: Option<bool>,
    resource_contents: HashMap<String, u64>,
    ship_contents: HashSet<String>,
    allowed: Option<(Vec<String>, Vec<String>)>,
    target: u64,
    capacity: u64,
    propagate: Option<bool>,
}

impl PluripotentStockpile {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
    ) -> internal::PluripotentStockpile {
        let stockpile = internal::PluripotentStockpile {
            visibility: self.visibility.unwrap_or(true),
            resource_contents: self
                .resource_contents
                .drain()
                .map(|(id, num)| (*resourceidmap.get(&id).expect("Resource is invalid!"), num))
                .collect(),
            //NOTE: This just gives a pluripotent stockpile an empty ship_contents because IIRC there's not a way to actually create shipinstances in the json
            ship_contents: HashSet::new(),
            allowed: self.allowed.map(|(resources, shipclasses)| {
                (
                    resources
                        .iter()
                        .map(|id| *resourceidmap.get(id).expect("Resource is invalid!"))
                        .collect(),
                    shipclasses
                        .iter()
                        .map(|id| *shipclassidmap.get(id).expect("Shipclass is invalid!"))
                        .collect(),
                )
            }),
            target: self.target,
            capacity: self.capacity,
            propagate: self.propagate.unwrap_or(true),
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
    launch_volume: u64, //how much volume the hangar can launch at one time in battle
    launch_interval: u64, //time between launches in battle
    propagate: bool, //whether or not hangar generates saliences
}

impl HangarClass {
    fn hydrate(
        mut self,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
        hangarclassidmap: &HashMap<String, internal::Key<internal::HangarClass>>,
    ) -> internal::HangarClass {
        let hangarclass = internal::HangarClass {
            id: *hangarclassidmap.get(&self.id).unwrap(),
            visiblename: self.visiblename,
            description: self.description,
            visibility: self.visibility.unwrap_or(true),
            capacity: self.capacity,
            target: self.target,
            allowed: self
                .allowed
                .drain(0..)
                .map(|x| *shipclassidmap.get(&x).unwrap())
                .collect(),
            ideal: self
                .ideal
                .drain()
                .map(|(k, v)| (*shipclassidmap.get(&k).unwrap(), v))
                .collect(),
            launch_volume: self.launch_volume,
            launch_interval: self.launch_interval,
            propagate: self.propagate,
        };
        hangarclass
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EngineClass {
    id: String,
    visiblename: String,
    description: String,
    basehealth: Option<u64>,
    toughnessscalar: f32,
    visibility: Option<bool>,
    inputs: Vec<UnipotentResourceStockpile>,
    forbidden_nodeflavors: Option<Vec<String>>,
    forbidden_edgeflavors: Option<Vec<String>>,
    speed: u64,    //number of edges the engine allows a ship to traverse when used
    cooldown: u64, //number of turns engine must wait before being used again
}

impl EngineClass {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        nodeflavoridmap: &HashMap<String, internal::Key<internal::NodeFlavor>>,
        edgeflavoridmap: &HashMap<String, internal::Key<internal::EdgeFlavor>>,
        engineclassidmap: &HashMap<String, internal::Key<internal::EngineClass>>,
    ) -> internal::EngineClass {
        internal::EngineClass {
            id: *engineclassidmap.get(&self.id).unwrap(),
            visiblename: self.visiblename,
            description: self.description,
            basehealth: self.basehealth,
            toughnessscalar: self.toughnessscalar,
            visibility: self.visibility.unwrap_or(true),
            inputs: self
                .inputs
                .drain(0..)
                .map(|x| x.hydrate(resourceidmap))
                .collect(),
            forbidden_nodeflavors: match self.forbidden_nodeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| *nodeflavoridmap.get(&s).unwrap())
                    .collect(),
                None => Vec::new(),
            },
            forbidden_edgeflavors: match self.forbidden_edgeflavors {
                Some(mut v) => v
                    .drain(0..)
                    .map(|s| *edgeflavoridmap.get(&s).unwrap())
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
    inputs: Vec<UnipotentResourceStockpile>,
    repair_points: i64,
    repair_factor: f32,
    engine_repair_points: i64,
    engine_repair_factor: f32,
}

impl RepairerClass {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        repairerclassidmap: &HashMap<String, internal::Key<internal::RepairerClass>>,
    ) -> internal::RepairerClass {
        let repairerclass = internal::RepairerClass {
            id: *repairerclassidmap.get(&self.id).unwrap(),
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
    inputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset consumption
    outputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        factoryclassidmap: &HashMap<String, internal::Key<internal::FactoryClass>>,
    ) -> internal::FactoryClass {
        let factoryclass = internal::FactoryClass {
            id: *factoryclassidmap.get(&self.id).unwrap(),
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
    inputs: Vec<UnipotentResourceStockpile>,
    outputs: HashMap<String, u64>,
    constructrate: u64,
    efficiency: f64,
}

impl ShipyardClass {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
        shipyardclassidmap: &HashMap<String, internal::Key<internal::ShipyardClass>>,
    ) -> internal::ShipyardClass {
        let shipyardclass = internal::ShipyardClass {
            id: *shipyardclassidmap.get(&self.id).unwrap(),
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
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
    ) -> (String, internal::ShipAI) {
        let shipai = internal::ShipAI {
            ship_attract_specific: self.ship_attract_specific,
            ship_attract_generic: self.ship_attract_generic,
            ship_cargo_attract: self
                .ship_cargo_attract
                .iter()
                .map(|(stringid, v)| (*shipclassidmap.get(stringid).unwrap(), *v))
                .collect(),
            resource_attract: self
                .resource_attract
                .iter()
                .map(|(stringid, v)| (*resourceidmap.get(stringid).unwrap(), *v))
                .collect(),
        };
        (self.id, shipai)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ShipClass {
    id: String,
    visiblename: String,
    description: String,
    basehull: u64,     //how many hull hitpoints this ship has by default
    basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    visibility: Option<bool>,
    hangarvol: Option<u64>, //how much hangar space this ship takes up when carried by a host
    cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
    stockpiles: Option<Vec<PluripotentStockpile>>,
    defaultweapons: Option<HashMap<String, u64>>, //a strikecraft's default weapons, which it always has with it
    hangars: Option<Vec<String>>,
    engines: Option<Vec<String>>,
    repairers: Option<Vec<String>>,
    factoryclasslist: Option<Vec<String>>,
    shipyardclasslist: Option<Vec<String>>,
    aiclass: String,
    defectchance: Option<HashMap<String, f64>>,
    toughnessscalar: Option<f32>,
    escapescalar: Option<f32>,
}

impl ShipClass {
    fn hydrate(
        self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        hangarclassidmap: &HashMap<String, internal::Key<internal::HangarClass>>,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
        engineclassidmap: &HashMap<String, internal::Key<internal::EngineClass>>,
        repairerclassidmap: &HashMap<String, internal::Key<internal::RepairerClass>>,
        factoryclassidmap: &HashMap<String, internal::Key<internal::FactoryClass>>,
        shipyardclassidmap: &HashMap<String, internal::Key<internal::ShipyardClass>>,
        shipaiidmap: &HashMap<String, internal::Key<internal::ShipAI>>,
        factionidmap: &HashMap<String, internal::Key<internal::Faction>>,
    ) -> internal::ShipClass {
        let shipclass = internal::ShipClass {
            id: *shipclassidmap.get(&self.id).unwrap(),
            visiblename: self.visiblename,
            description: self.description,
            basehull: self.basehull,
            basestrength: self.basestrength,
            visibility: self.visibility.unwrap_or(true),
            defaultweapons: self.defaultweapons.map(|map| {
                map.iter()
                    .map(|(id, n)| {
                        (
                            *resourceidmap
                                .get(id)
                                .unwrap_or_else(|| panic!("{} is not found!", id)),
                            *n,
                        )
                    })
                    .collect()
            }),
            hangarvol: self.hangarvol,
            cargovol: self.cargovol,
            stockpiles: self
                .stockpiles
                .unwrap_or(Vec::new())
                .iter()
                .map(|stockpile| stockpile.clone().hydrate(&resourceidmap, &shipclassidmap))
                .collect(),
            hangars: self
                .hangars
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    *hangarclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                })
                .collect(),
            engines: self
                .engines
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    *engineclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                })
                .collect(),
            repairers: self
                .repairers
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    *repairerclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                })
                .collect(),
            factoryclasslist: self
                .factoryclasslist
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    *factoryclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                })
                .collect(),
            shipyardclasslist: self
                .shipyardclasslist
                .unwrap_or(Vec::new())
                .iter()
                .map(|id| {
                    *shipyardclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found!", id))
                })
                .collect(),
            aiclass: *shipaiidmap.get(&self.aiclass).unwrap(),
            defectchance: self
                .defectchance
                .unwrap_or(HashMap::new())
                .iter()
                .map(|(k, v)| (*factionidmap.get(k).unwrap(), *v))
                .collect(),
            toughnessscalar: self.toughnessscalar.unwrap_or(1.0),
            escapescalar: self.escapescalar.unwrap_or(1.0),
        };
        shipclass
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FleetClass {
    id: String,
    visiblename: String,
    description: String,
    strengthmod: (f32, u64),
    visibility: Option<bool>,
    fleetconfig: HashMap<String, u64>,
    defectchance: HashMap<String, f64>,
    disbandthreshold: f32,
}

impl FleetClass {
    fn hydrate(
        self,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
        fleetclassidmap: &HashMap<String, internal::Key<internal::FleetClass>>,
        factionidmap: &HashMap<String, internal::Key<internal::Faction>>,
    ) -> internal::FleetClass {
        let fleetclass = internal::FleetClass {
            id: *fleetclassidmap.get(&self.id).unwrap(),
            visiblename: self.visiblename,
            description: self.description,
            strengthmod: self.strengthmod,
            visibility: self.visibility.unwrap_or(true),
            fleetconfig: self
                .fleetconfig
                .iter()
                .map(|(stringid, n)| (*shipclassidmap.get(stringid).unwrap(), *n))
                .collect(),
            defectchance: self
                .defectchance
                .iter()
                .map(|(stringid, n)| (*factionidmap.get(stringid).unwrap(), *n))
                .collect(),
            disbandthreshold: self.disbandthreshold,
        };
        fleetclass
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

#[derive(Debug, Clone, Serialize, Deserialize)] //structure for modder-defined json
pub struct Root {
    nodeflavors: Vec<NodeFlavor>,
    systems: Vec<System>,
    edgeflavors: Vec<EdgeFlavor>,
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
    shipclasses: Vec<ShipClass>,
    fleetclasses: Vec<FleetClass>,
    battlescalars: BattleScalars,
}

impl Root {
    //hydration method
    pub fn hydrate(mut self) -> internal::Root {
        //here we iterate over the json systems to create a map between nodes' json string-ids and internal ids
        let nodeidmap: HashMap<String, internal::Key<internal::Node>> = self
            .systems
            .iter()
            .flat_map(|system| system.nodes.iter())
            .enumerate()
            .map(|(i, node)| {
                (
                    node.id.clone(),
                    internal::Key::<internal::Node>::new_from_index(i),
                )
            })
            .collect();

        let edgeflavoridmap: HashMap<String, internal::Key<internal::EdgeFlavor>> = self
            .edgeflavors
            .iter()
            .enumerate()
            .map(|(i, edgeflavor)| (edgeflavor.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let edgeflavors: Vec<internal::EdgeFlavor> = self
            .edgeflavors
            .drain(0..)
            .enumerate()
            .map(|(i, edgeflavor)| {
                //we turn the enumeration index into a key
                let id = internal::Key::new_from_index(i);
                //we make sure the key we just made matches the nodeflavor's entry in the idmap
                assert_eq!(id, *edgeflavoridmap.get(&edgeflavor.id).unwrap());
                edgeflavor.hydrate()
            })
            .collect();

        //here we convert the json edge list into a set of pairs of internal node ids
        let mut edges: HashMap<
            (internal::Key<internal::Node>, internal::Key<internal::Node>),
            internal::Key<internal::EdgeFlavor>,
        > = self
            .edges
            .iter()
            .map(|(a, b, f)| {
                let aid = *nodeidmap.get(a).unwrap();
                let bid = *nodeidmap.get(b).unwrap();
                assert_ne!(aid, bid);
                (
                    (aid.min(bid), aid.max(bid)),
                    *edgeflavoridmap.get(f).unwrap(),
                )
            })
            .collect();

        let mut jsonnodes: Vec<Node> = Vec::new();

        //here we convert json systems into internal systems, and create a map between json string-id and internal id
        let (systems, systemidmap): (
            Vec<internal::System>,
            HashMap<String, internal::Key<internal::System>>,
        ) = self
            .systems
            .drain(0..)
            .enumerate()
            .map(|(i, system)| {
                //we hydrate the system, getting the system's stringid, the internal system struct, and a vec of the nodes that are in this system
                let (stringid, internalsystem, mut nodes) = system.hydrate(&nodeidmap);
                let mut nodeids: Vec<internal::Key<internal::Node>> = Vec::new();
                //here we build all-to-all edges between the nodes in the system
                nodes.iter().for_each(|node| {
                    let nodeid = *nodeidmap.get(&node.id).unwrap();
                    //we check whether the node is orphaned, and if it is we don't build any edges for it
                    //NOTE: This is only half-built and only works some of the time!
                    //We need to do a second check on the inner iteration, but I don't understand what it's doing well enough.
                    if node.orphan != Some(true) {
                        //we get the node's id from the id map
                        //we iterate over the nodeids, ensure that there aren't any duplicates, and push each pair of nodeids into edges
                        nodeids.iter().for_each(|rhs| {
                            if nodes
                                .iter()
                                .find(|n| nodeidmap.get(&n.id).unwrap() == rhs)
                                .unwrap()
                                .orphan
                                != Some(true)
                            {
                                assert_ne!(nodeid, *rhs, "Same node ID appears twice.");
                                edges.insert(
                                    (nodeid.min(*rhs), nodeid.max(*rhs)),
                                    internal::Key::new_from_index(0),
                                );
                            }
                        })
                    };
                    nodeids.push(nodeid);
                });

                //jsonnodes gets used later to generate the nodes vec, which we then turn into the internal table
                nodes.drain(0..).for_each(|node| {
                    jsonnodes.push(node);
                });
                //we create a system id from the enumeration index, then pair it with the system's stringid
                let kv = (
                    stringid,
                    internal::Key::<internal::System>::new_from_index(i),
                );
                (internalsystem, kv)
            })
            .unzip();

        let neighbors: HashMap<internal::Key<internal::Node>, Vec<internal::Key<internal::Node>>> =
            edges.iter().fold(HashMap::new(), |mut acc, (&(a, b), _)| {
                acc.entry(a).or_insert_with(Vec::new).push(b);
                acc.entry(b).or_insert_with(Vec::new).push(a);
                acc
            });

        let nodeflavoridmap: HashMap<String, internal::Key<internal::NodeFlavor>> = self
            .nodeflavors
            .iter()
            .enumerate()
            .map(|(i, nodeflavor)| (nodeflavor.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let nodeflavors: Vec<internal::NodeFlavor> = self
            .nodeflavors
            .drain(0..)
            .enumerate()
            .map(|(i, nodeflavor)| {
                //we turn the enumeration index into a key
                let id = internal::Key::new_from_index(i);
                //we make sure the key we just made matches the nodeflavor's entry in the idmap
                assert_eq!(id, *nodeflavoridmap.get(&nodeflavor.id).unwrap());
                nodeflavor.hydrate()
            })
            .collect();

        let factionidmap: HashMap<String, internal::Key<internal::Faction>> = self
            .factions
            .iter()
            .enumerate()
            .map(|(i, faction)| {
                let stringid = faction.id.clone();
                let kv_pair = (stringid, internal::Key::new_from_index(i));
                kv_pair
            })
            .collect();

        //fairly simple hydration process
        let factions: Vec<internal::Faction> = self
            .factions
            .drain(0..)
            .enumerate()
            .map(|(i, faction)| {
                //we turn the enumeration index into a key
                let id = internal::Key::new_from_index(i);
                //we make sure the key we just made matches the faction's entry in the idmap NOTE: Wait, why do we do this? The ID doesn't seem to get used anywhere. Couldn't we do this in any order?
                assert_eq!(id, *factionidmap.get(&faction.id).unwrap());
                faction.hydrate(&factionidmap)
            })
            .collect();

        let wars: HashSet<(
            internal::Key<internal::Faction>,
            internal::Key<internal::Faction>,
        )> = self
            .wars
            .iter()
            .map(|(a, b)| {
                let aid = *factionidmap.get(a).unwrap();
                let bid = *factionidmap.get(b).unwrap();
                assert_ne!(aid, bid);
                (aid.min(bid), aid.max(bid))
            })
            .collect();

        //same sort of deal here
        let (resourceidmap, resources): (
            HashMap<String, internal::Key<internal::Resource>>,
            Vec<internal::Resource>,
        ) = self
            .resources
            .drain(0..)
            .enumerate()
            .map(|(i, resource)| {
                let (stringid, internal_resource) = resource.hydrate();
                let kv_pair = (stringid, internal::Key::new_from_index(i));
                (kv_pair, internal_resource)
            })
            .unzip();

        let engineclassidmap: HashMap<String, internal::Key<internal::EngineClass>> = self
            .engineclasses
            .iter()
            .enumerate()
            .map(|(i, engineclass)| (engineclass.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let engineclasses: Vec<internal::EngineClass> = self
            .engineclasses
            .drain(0..)
            .map(|engineclass| {
                engineclass.hydrate(
                    &resourceidmap,
                    &nodeflavoridmap,
                    &edgeflavoridmap,
                    &engineclassidmap,
                )
            })
            .collect();

        let repairerclassidmap: HashMap<String, internal::Key<internal::RepairerClass>> = self
            .repairerclasses
            .iter()
            .enumerate()
            .map(|(i, repairerclass)| (repairerclass.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let repairerclasses: Vec<internal::RepairerClass> = self
            .repairerclasses
            .drain(0..)
            .map(|repairerclass| repairerclass.hydrate(&resourceidmap, &repairerclassidmap))
            .collect();

        let factoryclassidmap: HashMap<String, internal::Key<internal::FactoryClass>> = self
            .factoryclasses
            .iter()
            .enumerate()
            .map(|(i, factoryclass)| (factoryclass.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let factoryclasses: Vec<internal::FactoryClass> = self
            .factoryclasses
            .drain(0..)
            .map(|factoryclass| factoryclass.hydrate(&resourceidmap, &factoryclassidmap))
            .collect();

        //this is a dummy ship class, which is here so that salience processes that require a shipclass to be specified can be parsed correctly
        let generic_demand_ship = ShipClass {
            id: "generic_demand_ship".to_string(),
            visiblename: "Generic Demand Ship".to_string(),
            description: "".to_string(),
            basehull: 1,     //how many hull hitpoints this ship has by default
            basestrength: 0, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
            visibility: Some(false),
            aiclass: "basicai".to_string(), //aiclass
            defaultweapons: None, //a strikecraft's default weapons, which it always has with it
            hangarvol: None,      //how much hangar space this ship takes up when carried by a host
            cargovol: None, //how much cargo space this ship takes up when transported by a cargo ship
            factoryclasslist: None,
            shipyardclasslist: None,
            stockpiles: None,
            hangars: None,
            engines: None,
            repairers: None,
            defectchance: None,
            toughnessscalar: None,
            escapescalar: None,
        };

        //here we create the shipclassidmap, put the dummy ship class inside it, and then insert all the actual ship classes
        let shipclassidmap: HashMap<String, internal::Key<internal::ShipClass>> =
            iter::once(&generic_demand_ship)
                .chain(self.shipclasses.iter())
                .enumerate()
                .map(|(i, shipclass)| (shipclass.id.clone(), internal::Key::new_from_index(i)))
                .collect();

        let hangarclassidmap = self
            .hangarclasses
            .iter()
            .enumerate()
            .map(|(i, hangarclass)| (hangarclass.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let hangarclasses = self
            .hangarclasses
            .drain(0..)
            .map(|hangarclass| hangarclass.hydrate(&shipclassidmap, &hangarclassidmap))
            .collect();

        let shipyardclassidmap: HashMap<String, internal::Key<internal::ShipyardClass>> = self
            .shipyardclasses
            .iter()
            .enumerate()
            .map(|(i, shipyardclass)| (shipyardclass.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let shipyardclasses: Vec<internal::ShipyardClass> = self
            .shipyardclasses
            .drain(0..)
            .map(|shipyardclass| {
                shipyardclass.hydrate(&resourceidmap, &shipclassidmap, &shipyardclassidmap)
            })
            .collect();

        //this is probably going to be messed with a bunch when we switch nodes over to the key system so I'm not going to bother commenting it yet
        let nodes: Vec<internal::Node> = jsonnodes
            .drain(0..)
            .enumerate()
            .map(|(i, node)| {
                let (stringid, node) = node.hydrate(
                    &nodeflavoridmap,
                    &factionidmap,
                    &internal::Table::from_vec(factoryclasses.clone()),
                    &factoryclassidmap,
                    &internal::Table::from_vec(shipyardclasses.clone()),
                    &shipyardclassidmap,
                );
                assert_eq!(
                    *nodeidmap.get(&stringid).unwrap(),
                    internal::Key::<internal::Node>::new_from_index(i)
                );
                node
            })
            .collect();

        //same as with shipyard classes
        let (shipaiidmap, shipais): (
            HashMap<String, internal::Key<internal::ShipAI>>,
            Vec<internal::ShipAI>,
        ) = self
            .shipais
            .drain(0..)
            .enumerate()
            .map(|(i, shipai)| {
                let (stringid, internal_shipai) = shipai.hydrate(&resourceidmap, &shipclassidmap);
                let kv_pair = (stringid, internal::Key::new_from_index(i));
                (kv_pair, internal_shipai)
            })
            .unzip();

        //we hydrate shipclasses, starting with the generic demand ship
        let shipclasses: Vec<internal::ShipClass> = iter::once(generic_demand_ship)
            .chain(self.shipclasses.drain(0..))
            .map(|shipclass| {
                shipclass.hydrate(
                    &resourceidmap,
                    &hangarclassidmap,
                    &shipclassidmap,
                    &engineclassidmap,
                    &repairerclassidmap,
                    &factoryclassidmap,
                    &shipyardclassidmap,
                    &shipaiidmap,
                    &factionidmap,
                )
            })
            .collect();

        let fleetclassidmap: HashMap<String, internal::Key<internal::FleetClass>> = self
            .fleetclasses
            .iter()
            .enumerate()
            .map(|(i, fleetclass)| (fleetclass.id.clone(), internal::Key::new_from_index(i)))
            .collect();

        let fleetclasses: Vec<internal::FleetClass> = self
            .fleetclasses
            .drain(0..)
            .enumerate()
            .map(|(i, fleetclass)| {
                //we turn the enumeration index into a key
                let id = internal::Key::new_from_index(i);
                //we make sure the key we just made matches the fleetclass's entry in the idmap
                assert_eq!(id, *fleetclassidmap.get(&fleetclass.id).unwrap());
                let internal_fleetclass =
                    fleetclass.hydrate(&shipclassidmap, &fleetclassidmap, &factionidmap);
                internal_fleetclass
            })
            .collect();

        let battlescalars = self.battlescalars.hydrate();

        internal::Root {
            nodeflavors: internal::Table::from_vec(nodeflavors),
            nodes: internal::Table::from_vec(nodes),
            systems: internal::Table::from_vec(systems),
            edgeflavors: internal::Table::from_vec(edgeflavors),
            edges,
            neighbors,
            factions: internal::Table::from_vec(factions),
            wars,
            resources: internal::Table::from_vec(resources),
            hangarclasses: internal::Table::from_vec(hangarclasses),
            engineclasses: internal::Table::from_vec(engineclasses),
            repairerclasses: internal::Table::from_vec(repairerclasses),
            factoryclasses: internal::Table::from_vec(factoryclasses),
            shipyardclasses: internal::Table::from_vec(shipyardclasses),
            shipais: internal::Table::from_vec(shipais),
            shipclasses: internal::Table::from_vec(shipclasses),
            shipinstances: internal::Table::new(),
            shipinstancecounter: 0_usize,
            fleetclasses: internal::Table::from_vec(fleetclasses),
            fleetinstances: internal::Table::new(),
            engagements: internal::Table::new(),
            battlescalars,
            globalsalience: internal::GlobalSalience {
                resourcesalience: Vec::new(),
                shipclasssalience: Vec::new(),
                factionsalience: Vec::new(),
            },
            turn: 0_u64,
        }
    }
}
