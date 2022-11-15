//this is the section of the program that manages the json files defined by the modder
use crate::internal;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer_pretty};
use std::collections::{HashMap, HashSet};
use std::iter;

#[derive(Debug, Clone, Serialize, Deserialize)] //structure for modder-defined json
pub struct Root {
    systems: Vec<System>,
    nodeflavors: Vec<NodeFlavor>,
    links: Vec<(String, String)>,
    factions: Vec<Faction>,
    factoryclasses: Vec<FactoryClass>,
    shipyardclasses: Vec<ShipyardClass>,
    resources: Vec<Resource>,
    shipclasses: Vec<ShipClass>,
    shipais: Vec<ShipAI>,
    fleetclasses: Vec<FleetClass>,
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

        //here we convert the json edge list into a set of pairs of internal node ids
        let mut edges: HashSet<(internal::Key<internal::Node>, internal::Key<internal::Node>)> =
            self.links
                .iter()
                .map(|(a, b)| {
                    let aid = *nodeidmap.get(a).unwrap();
                    let bid = *nodeidmap.get(b).unwrap();
                    assert_ne!(aid, bid);
                    (aid.min(bid), aid.max(bid))
                })
                .collect();

        let mut jsonnodes: Vec<Node> = Vec::new();

        //here we convert json systems into internal systems, and create a map between json string-id and internal id
        let (systems, systemidmap): (Vec<internal::System>, HashMap<String, internal::SystemID>) =
            self.systems
                .drain(0..)
                .enumerate()
                .map(|(i, system)| {
                    //we hydrate the system, getting the system's stringid, the internal system struct, and a vec of the nodes that are in this system
                    let (stringid, internalsystem, mut nodes) = system.hydrate(&nodeidmap);
                    let mut nodeids: Vec<internal::Key<internal::Node>> = Vec::new();
                    //here we build all-to-all edges between the nodes in the system
                    nodes.iter().for_each(|node| {
                        //we get the node's id from the id map
                        let nodeid = *nodeidmap.get(&node.id).unwrap();
                        //we iterate over the nodeids, ensure that there aren't any duplicates, and push each pair of nodeids into edges
                        nodeids.iter().for_each(|&rhs| {
                            assert_ne!(nodeid, rhs, "Same node ID appears twice.");
                            edges.insert((nodeid.min(rhs), nodeid.max(rhs)));
                        });
                        nodeids.push(nodeid);
                    });

                    //NOTE: we turn the nodes vec into jsonnodes, for ?some reason?
                    nodes.drain(0..).for_each(|node| {
                        jsonnodes.push(node);
                    });
                    //we create a system id from the enumeration index, then pair it with the system's stringid
                    let kv = (stringid, internal::SystemID(i));
                    (internalsystem, kv)
                })
                .unzip();

        let neighbors: HashMap<internal::Key<internal::Node>, Vec<internal::Key<internal::Node>>> =
            edges.iter().fold(HashMap::new(), |mut acc, &(a, b)| {
                acc.entry(a).or_insert_with(Vec::new).push(b);
                acc.entry(b).or_insert_with(Vec::new).push(a);
                acc
            });

        let nodeflavoridmap: HashMap<String, internal::NodeFlavorID> = self
            .nodeflavors
            .iter()
            .enumerate()
            .map(|(i, nodeflavor)| (nodeflavor.id.clone(), internal::NodeFlavorID(i)))
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

        //let mut factions = internal::Table::new();
        //for faction in self.factions {
        //    factions.insert(faction)
        //}

        //let factionidmap: HashMap<String, internal::Key::<Faction>> = self.factions.iter().map(|faction|{
        //    let stringid = faction.id;
        //    let hydrated = faction.hydrate()
        //}).collect()

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
                let internal_faction = faction.hydrate(&factionidmap);
                internal_faction
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

        //pretty much exactly the same as resources
        let (factoryclassidmap, factoryclasses): (
            HashMap<String, internal::Key<internal::FactoryClass>>,
            Vec<internal::FactoryClass>,
        ) = self
            .factoryclasses
            .drain(0..)
            .enumerate()
            .map(|(i, factoryclass)| {
                let (stringid, internal_factoryclass) = factoryclass.hydrate(&resourceidmap);
                let kv_pair = (stringid, internal::Key::new_from_index(i));
                (kv_pair, internal_factoryclass)
            })
            .unzip();

        //this is a dummy ship class, which is here so that salience processes that require a shipclass to be specified can be parsed correctly
        let generic_demand_ship = ShipClass {
            id: "generic_demand_ship".to_string(),
            visiblename: "Generic Demand Ship".to_string(),
            description: "".to_string(),
            basehull: None,  //how many hull hitpoints this ship has by default
            basestrength: 0, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
            aiclass: "yes".to_string(),
            defaultweapons: None, //a strikecraft's default weapons, which it always has with it
            hangarcap: None,      //this ship's capacity for carrying active strikecraft
            weaponcap: None,      //this ship's capacity for carrying strikecraft weapons
            cargocap: None,       //this ship's capacity for carrying cargo
            hangarvol: None,      //how much hangar space this ship takes up when carried by a host
            cargovol: None, //how much cargo space this ship takes up when transported by a cargo ship
            factoryclasslist: Vec::new(),
            shipyardclasslist: Vec::new(),
            stockpiles: Vec::new(),
            hyperdrive: None, //number of links this ship can traverse in one turn
            compconfig: None, //ideal configuration for this ship's strikecraft complement
            defectchance: None,
        };

        //here we create the shipclassidmap, put the dummy ship class inside it, and then insert all the actual ship classes
        let shipclassidmap: HashMap<String, internal::Key<internal::ShipClass>> =
            iter::once(&generic_demand_ship)
                .chain(self.shipclasses.iter())
                .enumerate()
                .map(|(i, shipclass)| (shipclass.id.clone(), internal::Key::new_from_index(i)))
                .collect();

        let (shipyardclassidmap, shipyardclasses): (
            HashMap<String, internal::Key<internal::ShipyardClass>>,
            Vec<internal::ShipyardClass>,
        ) = self
            .shipyardclasses
            .drain(0..)
            .enumerate()
            .map(|(i, shipyardclass)| {
                //we hydrate the shipclass, returning both the stringid and the internal shipclass, then create a key and return all three things
                let (stringid, internal_shipyardclass) =
                    shipyardclass.hydrate(&resourceidmap, &shipclassidmap);
                let kv_pair = (stringid, internal::Key::new_from_index(i));
                (kv_pair, internal_shipyardclass)
            })
            .unzip();

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
                    &shipclassidmap,
                    &factoryclassidmap,
                    &shipyardclassidmap,
                    &shipaiidmap,
                )
            })
            .collect();

        //same as shipyard classes
        let (fleetclassidmap, fleetclasses): (
            HashMap<String, internal::Key<internal::FleetClass>>,
            Vec<internal::FleetClass>,
        ) = self
            .fleetclasses
            .drain(0..)
            .enumerate()
            .map(|(i, fleetclass)| {
                let (stringid, internal_fleetclass) = fleetclass.hydrate(&shipclassidmap);
                let kv_pair = (stringid, internal::Key::new_from_index(i));
                (kv_pair, internal_fleetclass)
            })
            .unzip();

        internal::Root {
            systems: internal::Table::from_vec(systems),
            nodes: internal::Table::from_vec(nodes),
            edges: edges,
            neighbors,
            factions: internal::Table::from_vec(factions),
            factoryclasses: internal::Table::from_vec(factoryclasses),
            shipyardclasses: internal::Table::from_vec(shipyardclasses),
            resources: internal::Table::from_vec(resources),
            shipais: internal::Table::from_vec(shipais),
            shipclasses: internal::Table::from_vec(shipclasses),
            shipinstances: internal::Table::new(),
            shipinstancecounter: 0_usize,
            fleetclasses: internal::Table::from_vec(fleetclasses),
            fleetinstances: internal::Table::new(),
            turn: 0_u64,
        }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Node {
    id: String,
    visiblename: String, //location name as shown to player
    distance: u64,       //node's distance from the sun, used for skybox generation
    description: String,
    flavor: String, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    factorylist: Vec<String>, //a list of the factories this node has, in the form of FactoryClass IDs
    shipyardlist: Vec<String>,
    environment: String, //name of the FRED environment to use for missions set in this node
    allegiance: String,  //faction that currently holds the node
    efficiency: Option<f64>, //efficiency of any production facilities in this node; changes over time based on faction ownership
}

impl Node {
    fn hydrate(
        self,
        nodeflavoridmap: &HashMap<String, internal::NodeFlavorID>,
        factionidmap: &HashMap<String, internal::Key<internal::Faction>>,
        factoryclasses: &internal::Table<internal::FactoryClass>,
        factoryclassidmap: &HashMap<String, internal::Key<internal::FactoryClass>>,
        shipyardclasses: &internal::Table<internal::ShipyardClass>,
        shipyardclassidmap: &HashMap<String, internal::Key<internal::ShipyardClass>>,
    ) -> (String, internal::Node) {
        let node = internal::Node {
            visiblename: self.visiblename,
            system: internal::SystemID(0),
            distance: self.distance,
            description: self.description,
            flavor: *nodeflavoridmap
                .get(&self.flavor)
                .expect("Node flavor field is not correctly defined!"),
            factoryinstancelist: self
                .factorylist
                .iter()
                .map(|stringid| {
                    let classid = factoryclassidmap.get(stringid).unwrap();
                    factoryclasses.get(*classid).instantiate(true)
                })
                .collect(),
            shipyardinstancelist: self
                .shipyardlist
                .iter()
                .map(|stringid| {
                    let classid = shipyardclassidmap
                        .get(stringid)
                        .expect(&format!("Shipyard '{}' does not exist.", stringid));
                    shipyardclasses.get(*classid).instantiate(true)
                })
                .collect(),
            environment: self.environment,
            allegiance: *factionidmap
                .get(&self.allegiance)
                .expect("Allegiance field is not correctly defined!"),
            efficiency: self.efficiency.unwrap_or(1.0),
            threat: factionidmap.values().map(|&id| (id, 0_f32)).collect(),
        };
        (self.id, node)
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq, Serialize, Deserialize)]
struct NodeFlavor {
    id: String,
    visiblename: String,
    description: String,
}

impl NodeFlavor {
    fn hydrate(self) -> (String, internal::NodeFlavor) {
        let nodeflavor = internal::NodeFlavor {
            visiblename: self.visiblename,
            description: self.description,
        };
        (self.id, nodeflavor)
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
            relations: factionidmap
                .iter()
                .map(|(name, id)| (*id, self.relations.get(name).map(|&x| x).unwrap_or(0_f32)))
                .collect(),
        };
        faction
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stockpile {
    resourcetype: String,
    contents: u64,
    rate: Option<u64>,
    target: u64,
    capacity: u64,
    propagate: Option<bool>,
}

impl Stockpile {
    fn hydrate(
        self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
    ) -> internal::UnipotentResourceStockpile {
        let stockpile = internal::UnipotentResourceStockpile {
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
struct FactoryClass {
    id: String,
    visiblename: String,
    description: String,
    inputs: Vec<Stockpile>,  //the data for the factory's asset consumption
    outputs: Vec<Stockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
    ) -> (String, internal::FactoryClass) {
        let factoryclass = internal::FactoryClass {
            visiblename: self.visiblename,
            description: self.description,
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
        (self.id, factoryclass)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShipyardClass {
    id: String,
    visiblename: Option<String>,
    description: Option<String>,
    inputs: Vec<Stockpile>,
    outputs: HashMap<String, u64>,
    constructrate: u64,
    efficiency: f64,
}

impl ShipyardClass {
    fn hydrate(
        mut self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
    ) -> (String, internal::ShipyardClass) {
        let shipyardclass = internal::ShipyardClass {
            visiblename: self.visiblename,
            description: self.description,
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
        (self.id, shipyardclass)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Resource {
    id: String,
    visiblename: String,
    description: String,
    cargovol: u64, //how much space a one unit of this resource takes up when transported by a cargo ship
    valuemult: u64, //how valuable the AI considers one unit of this resource to be
}

impl Resource {
    fn hydrate(self) -> (String, internal::Resource) {
        let resource = internal::Resource {
            visiblename: self.visiblename,
            description: self.description,
            cargovol: self.cargovol,
            valuemult: self.valuemult,
        };
        (self.id, resource)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ShipClass {
    id: String,
    visiblename: String,
    description: String,
    basehull: Option<u64>, //how many hull hitpoints this ship has by default
    basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    aiclass: String,
    defaultweapons: Option<HashMap<String, u64>>, //a strikecraft's default weapons, which it always has with it
    hangarcap: Option<u64>, //this ship's capacity for carrying active strikecraft
    weaponcap: Option<u64>, //this ship's capacity for carrying strikecraft weapons
    cargocap: Option<u64>,  //this ship's capacity for carrying cargo
    hangarvol: Option<u64>, //how much hangar space this ship takes up when carried by a host
    cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
    factoryclasslist: Vec<String>,
    shipyardclasslist: Vec<String>,
    stockpiles: Vec<Stockpile>,
    hyperdrive: Option<u64>, //number of links this ship can traverse in one turn
    compconfig: Option<HashMap<String, u64>>, //ideal configuration for this ship's strikecraft complement
    defectchance: Option<HashMap<String, f64>>,
}

impl ShipClass {
    fn hydrate(
        self,
        resourceidmap: &HashMap<String, internal::Key<internal::Resource>>,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
        factoryclassidmap: &HashMap<String, internal::Key<internal::FactoryClass>>,
        shipyardclassidmap: &HashMap<String, internal::Key<internal::ShipyardClass>>,
        shipaiidmap: &HashMap<String, internal::Key<internal::ShipAI>>,
    ) -> internal::ShipClass {
        let shipclass = internal::ShipClass {
            id: *shipclassidmap.get(&self.id).unwrap(),
            visiblename: self.visiblename,
            description: self.description,
            basehull: self.basehull,
            basestrength: self.basestrength,
            aiclass: *shipaiidmap.get(&self.aiclass).unwrap(),
            defaultweapons: self.defaultweapons.map(|map| {
                map.iter()
                    .map(|(id, n)| {
                        (
                            *resourceidmap
                                .get(id)
                                .unwrap_or_else(|| panic!("{} is not found", id)),
                            *n,
                        )
                    })
                    .collect()
            }),
            hangarcap: self.hangarcap,
            weaponcap: self.weaponcap,
            cargocap: self.cargocap,
            hangarvol: self.hangarvol,
            cargovol: self.cargovol,
            factoryclasslist: self
                .factoryclasslist
                .iter()
                .map(|id| {
                    *factoryclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found", id))
                })
                .collect(),
            shipyardclasslist: self
                .shipyardclasslist
                .iter()
                .map(|id| {
                    *shipyardclassidmap
                        .get(id)
                        .unwrap_or_else(|| panic!("{} is not found", id))
                })
                .collect(),
            stockpiles: self
                .stockpiles
                .iter()
                .map(|stockpile| stockpile.clone().hydrate(&resourceidmap))
                .collect(),
            hyperdrive: self.hyperdrive,
            compconfig: self.compconfig.map(|map| {
                map.iter()
                    .map(|(id, n)| (*shipclassidmap.get(id).unwrap(), *n))
                    .collect()
            }),
            defectchance: HashMap::new(),
        };
        shipclass
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FleetClass {
    id: String,
    visiblename: String,
    description: String,
    fleetconfig: HashMap<String, u64>,
}

impl FleetClass {
    fn hydrate(
        self,
        shipclassidmap: &HashMap<String, internal::Key<internal::ShipClass>>,
    ) -> (String, internal::FleetClass) {
        let fleetclass = internal::FleetClass {
            visiblename: self.visiblename,
            description: self.description,
            fleetconfig: self
                .fleetconfig
                .iter()
                .map(|(stringid, n)| (*shipclassidmap.get(stringid).unwrap(), *n))
                .collect(),
        };
        (self.id, fleetclass)
    }
}
