use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer_pretty};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::time::{Duration, Instant};

mod json {

    use crate::internal;

    use serde::de::Deserializer;
    use serde::{Deserialize, Serialize};
    use serde_json::{from_reader, to_writer_pretty};
    use std::collections::{HashMap, HashSet};

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
        fleetclasses: Vec<FleetClass>,
    }

    impl Root {
        //hydration method
        pub fn hydrate(mut self) -> internal::Root {
            let nodeidmap: HashMap<String, internal::NodeID> = self
                .systems
                .iter()
                .flat_map(|system| system.nodes.iter())
                .enumerate()
                .map(|(i, node)| (node.id.clone(), internal::NodeID(i)))
                .collect();

            let mut edges: HashSet<(internal::NodeID, internal::NodeID)> = self
                .links
                .iter()
                .map(|(a, b)| {
                    let aid = *nodeidmap.get(a).unwrap();
                    let bid = *nodeidmap.get(b).unwrap();
                    assert_ne!(aid, bid);
                    (aid.min(bid), aid.max(bid))
                })
                .collect();
            let mut jsonnodes: Vec<Node> = Vec::new();
            let (systems, systemidmap): (
                Vec<internal::System>,
                HashMap<String, internal::SystemID>,
            ) = self
                .systems
                .drain(0..)
                .enumerate()
                .map(|(i, system)| {
                    let (stringid, internalsystem, mut nodes) = system.hydrate(&nodeidmap);
                    let mut nodeids: Vec<internal::NodeID> = Vec::new();
                    nodes.iter().for_each(|node| {
                        let nodeid = *nodeidmap.get(&node.id).unwrap();
                        nodeids.iter().for_each(|&rhs| {
                            assert_ne!(nodeid, rhs, "Same node ID appears twice.");
                            edges.insert((nodeid.min(rhs), nodeid.max(rhs)));
                        });
                        nodeids.push(nodeid);
                    });

                    nodes.drain(0..).for_each(|node| {
                        jsonnodes.push(node);
                    });
                    let kv = (stringid, internal::SystemID(i));
                    (internalsystem, kv)
                })
                .unzip();

            let neighbors: HashMap<internal::NodeID, Vec<internal::NodeID>> =
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

            let factionidmap: HashMap<String, internal::FactionID> = self
                .factions
                .iter()
                .enumerate()
                .map(|(i, faction)| {
                    let stringid = faction.id.clone();
                    let kv_pair = (stringid, internal::FactionID(i));
                    kv_pair
                })
                .collect();

            let factions: Vec<internal::Faction> = self
                .factions
                .drain(0..)
                .enumerate()
                .map(|(i, faction)| {
                    let id = internal::FactionID(i);
                    assert_eq!(id, *factionidmap.get(&faction.id).unwrap());
                    let internal_faction = faction.hydrate(&factionidmap);
                    internal_faction
                })
                .collect();

            let (resourceidmap, resources): (
                HashMap<String, internal::ResourceID>,
                Vec<internal::Resource>,
            ) = self
                .resources
                .drain(0..)
                .enumerate()
                .map(|(i, resource)| {
                    let (stringid, internal_resource) = resource.hydrate();
                    let kv_pair = (stringid, internal::ResourceID(i));
                    (kv_pair, internal_resource)
                })
                .unzip();

            let (factoryclassidmap, factoryclasses): (
                HashMap<String, internal::FactoryClassID>,
                Vec<internal::FactoryClass>,
            ) = self
                .factoryclasses
                .drain(0..)
                .enumerate()
                .map(|(i, factoryclass)| {
                    let (stringid, internal_factoryclass) = factoryclass.hydrate(&resourceidmap);
                    let kv_pair = (stringid, internal::FactoryClassID(i));
                    (kv_pair, internal_factoryclass)
                })
                .unzip();

            let shipclassidmap: HashMap<String, internal::ShipClassID> = self
                .shipclasses
                .iter()
                .enumerate()
                .map(|(i, shipclass)| (shipclass.id.clone(), internal::ShipClassID(i)))
                .collect();
            let (shipyardclassidmap, shipyardclasses): (
                HashMap<String, internal::ShipyardClassID>,
                Vec<internal::ShipyardClass>,
            ) = self
                .shipyardclasses
                .drain(0..)
                .enumerate()
                .map(|(i, shipyardclass)| {
                    let (stringid, internal_shipyardclass) =
                        shipyardclass.hydrate(&resourceidmap, &shipclassidmap);
                    let kv_pair = (stringid, internal::ShipyardClassID(i));
                    (kv_pair, internal_shipyardclass)
                })
                .unzip();

            let nodes: Vec<internal::Node> = jsonnodes
                .drain(0..)
                .enumerate()
                .map(|(i, node)| {
                    let (stringid, node) = node.hydrate(
                        &nodeflavoridmap,
                        &factionidmap,
                        &factoryclasses,
                        &factoryclassidmap,
                        &shipyardclasses,
                        &shipyardclassidmap,
                    );
                    assert_eq!(*nodeidmap.get(&stringid).unwrap(), internal::NodeID(i));
                    node
                })
                .collect();

            let shipclasses: Vec<internal::ShipClass> = self
                .shipclasses
                .drain(0..)
                .map(|shipclass| {
                    shipclass.hydrate(
                        &resourceidmap,
                        &shipclassidmap,
                        &factoryclassidmap,
                        &shipyardclassidmap,
                    )
                })
                .collect();
            let (fleetclassidmap, fleetclasses): (
                HashMap<String, internal::FleetClassID>,
                Vec<internal::FleetClass>,
            ) = self
                .fleetclasses
                .drain(0..)
                .enumerate()
                .map(|(i, fleetclass)| {
                    let (stringid, internal_fleetclass) = fleetclass.hydrate(&shipclassidmap);
                    let kv_pair = (stringid, internal::FleetClassID(i));
                    (kv_pair, internal_fleetclass)
                })
                .unzip();

            internal::Root {
                systems,
                nodes,
                edges,
                neighbors,
                factions,
                factoryclasses,
                shipyardclasses,
                resources,
                shipclasses,
                shipinstances: HashMap::new(),
                shipinstancecounter: 0_usize,
                fleetclasses,
                fleetinstances: HashMap::new(),
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
            nodeidmap: &HashMap<String, internal::NodeID>,
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
            factionidmap: &HashMap<String, internal::FactionID>,
            factorylist: &Vec<internal::FactoryClass>,
            factoryclassidmap: &HashMap<String, internal::FactoryClassID>,
            shipyardlist: &Vec<internal::ShipyardClass>,
            shipyardclassidmap: &HashMap<String, internal::ShipyardClassID>,
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
                        factorylist[classid.0].instantiate(true)
                    })
                    .collect(),
                shipyardinstancelist: self
                    .shipyardlist
                    .iter()
                    .map(|stringid| {
                        let classid = shipyardclassidmap.get(stringid).unwrap();
                        shipyardlist[classid.0].instantiate(true)
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
        fn hydrate(self, factionidmap: &HashMap<String, internal::FactionID>) -> internal::Faction {
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
            resourceidmap: &HashMap<String, internal::ResourceID>,
        ) -> internal::Stockpile {
            let stockpile = internal::Stockpile {
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
            resourceidmap: &HashMap<String, internal::ResourceID>,
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
            resourceidmap: &HashMap<String, internal::ResourceID>,
            shipclassidmap: &HashMap<String, internal::ShipClassID>,
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
        defaultweapons: Option<HashMap<String, u64>>, //a strikecraft's default weapons, which it always has with it
        hangarcap: Option<u64>, //this ship's capacity for carrying active strikecraft
        weaponcap: Option<u64>, //this ship's capacity for carrying strikecraft weapons
        cargocap: Option<u64>,  //this ship's capacity for carrying cargo
        hangarvol: Option<u64>, //how much hangar space this ship takes up when carried by a host
        cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
        factoryclasslist: Vec<String>,
        shipyardclasslist: Vec<String>,
        hyperdrive: Option<u64>, //number of links this ship can traverse in one turn
        compconfig: Option<HashMap<String, u64>>, //ideal configuration for this ship's strikecraft complement
        defectchance: Option<HashMap<String, f64>>,
    }

    impl ShipClass {
        fn hydrate(
            self,
            resourceidmap: &HashMap<String, internal::ResourceID>,
            shipclassidmap: &HashMap<String, internal::ShipClassID>,
            factoryclassidmap: &HashMap<String, internal::FactoryClassID>,
            shipyardclassidmap: &HashMap<String, internal::ShipyardClassID>,
        ) -> internal::ShipClass {
            let shipclass = internal::ShipClass {
                id: *shipclassidmap.get(&self.id).unwrap(),
                visiblename: self.visiblename,
                description: self.description,
                basehull: self.basehull,
                basestrength: self.basestrength,
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
    struct FleetClass {
        id: String,
        visiblename: String,
        description: String,
        fleetconfig: HashMap<String, u64>,
    }

    impl FleetClass {
        fn hydrate(
            self,
            shipclassidmap: &HashMap<String, internal::ShipClassID>,
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
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

mod internal {

    use std::collections::{HashMap, HashSet};

    #[derive(Debug)]
    pub struct Root {
        pub systems: Vec<System>,
        pub nodes: Vec<Node>,
        pub edges: HashSet<(NodeID, NodeID)>,
        pub neighbors: HashMap<NodeID, Vec<NodeID>>,
        pub factions: Vec<Faction>,
        pub factoryclasses: Vec<FactoryClass>,
        pub shipyardclasses: Vec<ShipyardClass>,
        pub resources: Vec<Resource>,
        pub shipclasses: Vec<ShipClass>,
        pub shipinstances: HashMap<ShipInstanceID, ShipInstance>,
        pub shipinstancecounter: usize,
        pub fleetclasses: Vec<FleetClass>,
        pub fleetinstances: HashMap<FleetInstanceID, FleetInstance>,
        pub turn: u64,
    }

    impl Root {
        pub fn process_turn(&mut self) {
            self.nodes
                .iter_mut()
                .enumerate()
                .for_each(|(nodeindex, node)| {
                    let nodeid = NodeID(nodeindex);
                    node.factoryinstancelist
                        .iter_mut()
                        .for_each(|factory| factory.process(node.efficiency));
                });
            let ship_plan_list: Vec<(ShipClassID, ShipLocationFlavor, FactionID)> = self
                .nodes
                .iter_mut()
                .enumerate()
                .map(|(nodeindex, node)| {
                    let nodeid = NodeID(nodeindex);
                    node.shipyardinstancelist
                        .iter_mut()
                        .map(|shipyard| {
                            let ship_plans =
                                shipyard.plan_ships(node.efficiency, &self.shipclasses);
                            ship_plans
                                .iter()
                                .map(|&ship_plan| {
                                    (ship_plan, ShipLocationFlavor::Node(nodeid), node.allegiance)
                                })
                                .collect::<Vec<_>>()
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect();
            let n_newships = ship_plan_list
                .iter()
                .map(|&(id, location, faction)| self.create_ship(id, location, faction))
                .count();
            println!("Built {} new ships.", n_newships);
            self.turn += 1;
            println!("It is now turn {}.", self.turn);
        }
        fn create_ship(
            &mut self,
            class_id: ShipClassID,
            location: ShipLocationFlavor,
            faction: FactionID,
        ) -> ShipInstanceID {
            let new_ship = self.shipclasses[class_id.0].instantiate(
                location,
                faction,
                &self.factoryclasses,
                &self.shipyardclasses,
            );
            self.shipinstancecounter += 1;
            let ship_instance_id = ShipInstanceID(self.shipinstancecounter);
            if self
                .shipinstances
                .insert(ship_instance_id, new_ship)
                .is_some()
            {
                panic!(
                    "{:?} tried to occupy same ID as existing ship!",
                    ship_instance_id
                );
            }
            ship_instance_id
        }

        fn get_node_strength(&self, nodeid: NodeID, faction: FactionID) -> u64 {
            self.shipinstances
                .values()
                .filter(|ship| ship.get_node(&self.shipinstances, &self.fleetinstances) == nodeid)
                .filter(|ship| ship.allegiance == faction)
                .map(|ship| ship.strength)
                .sum()
        }

        pub fn calculate_values<S: Salience<P> + Copy, P: Polarity>(
            &self,
            salience: S,
            factionid: FactionID,
            n_iters: usize,
        ) -> Vec<f32> {
            //Length equals nodes owned by faction and producing specified salience
            let node_initial_salience_map: Vec<(NodeID, f32)> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.allegiance == factionid)
                .filter_map(|(i, node)| {
                    let id = NodeID(i);
                    salience
                        .get_value((id, node), &self.shipinstances, &self.fleetinstances)
                        .map(|v| (id, v))
                })
                .collect();
            //Length equals all nodes
            let tagged_threats: Vec<HashMap<FactionID, f32>> = self
                .nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let id = NodeID(i);
                    node.threat
                        .iter()
                        .map(|(f, t)| {
                            let value = t * self.factions[factionid.0].relations.get(f).unwrap();
                            (*f, value)
                        })
                        .collect()
                })
                .collect();
            //Length equals all nodes
            let node_degradations: Vec<f32> = tagged_threats
                .iter()
                .map(|map| {
                    let sum = map.values().sum();
                    scale_from_threat(sum, 20_f32) * S::DEG_MULT * 0.8
                })
                .collect();
            //Outer vec length equals all nodes; inner vec equals nodes owned by faction and producing specified salience -- but only the instance of self-node contains a nonzero value
            let node_salience_state: Vec<Vec<f32>> = self
                .nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    let id = NodeID(i);
                    node_initial_salience_map
                        .iter()
                        .map(|&(sourcenodeid, value)| value * ((sourcenodeid == id) as u8) as f32)
                        .collect()
                })
                .collect();

            let n_tags = node_initial_salience_map.len();
            let node_salience_state = (0..n_iters).fold(node_salience_state, |mut state, n_iter| {
                println!("Completed {} iterations of salience propagation.", n_iter);
                self.edges.iter().for_each(|(a, b)| {
                    let deg_a = node_degradations[a.0];
                    let deg_b = node_degradations[b.0];
                    for i in 0..n_tags {
                        state[a.0][i] = state[a.0][i].max(state[b.0][i] * deg_b);
                        state[b.0][i] = state[b.0][i].max(state[a.0][i] * deg_a);
                    }
                });
                state
            });
            node_salience_state
                .iter()
                .map(|salience| salience.iter().sum())
                .collect()
        }

        pub fn update_node_threats(&mut self, n_steps:usize){
            self.factions()
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
    }

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

    trait Polarity {}

    pub mod polarity {

        use super::Polarity;

        pub struct Supply {}

        impl Polarity for Supply {}

        pub struct Demand {}

        impl Polarity for Demand {}
    }

    trait Salience<P: Polarity> {
        const DEG_MULT: f32;
        fn get_value(
            self,
            node: (NodeID, &Node),
            shipinstances: &HashMap<ShipInstanceID, ShipInstance>,
            fleetinstances: &HashMap<FleetInstanceID, FleetInstance>,
        ) -> Option<f32>;
    }

    impl Salience<polarity::Supply> for FactionID {
        const DEG_MULT: f32 = 0.5;
        fn get_value(
            self,
            (nodeid, node): (NodeID, &Node),
            shipinstances: &HashMap<ShipInstanceID, ShipInstance>,
            fleetinstances: &HashMap<FleetInstanceID, FleetInstance>,
        ) -> Option<f32> {
            let node_strength:u64 = shipinstances
            .values()
            .filter(|ship| ship.get_node(shipinstances, fleetinstances) == nodeid)
            .filter(|ship| ship.allegiance == self)
            .map(|ship| ship.strength)
            .sum();
            Some(node_strength)
                .filter(|&strength| strength != 0)
                .map(|strength| strength as f32)
        }
    }

    impl Salience<polarity::Supply> for ResourceID {
        const DEG_MULT: f32 = 1.0;
        fn get_value(
            self,
            (nodeid, node): (NodeID, &Node),
            shipinstances: &HashMap<ShipInstanceID, ShipInstance>,
            fleetinstances: &HashMap<FleetInstanceID, FleetInstance>,
        ) -> Option<f32> {
            let factorysupply: u64 = node
                .factoryinstancelist
                .iter()
                .map(|factory| {
                    factory
                        .outputs
                        .iter()
                        .filter(|output| (output.resourcetype == self) & (output.propagate == true))
                        .map(|output| output.contents)
                        .sum::<u64>()
                })
                .sum::<u64>();
            let shipsupply: u64 = shipinstances
                .values()
                .filter(|ship| ship.get_node(&shipinstances, &fleetinstances) == nodeid)
                .map(|ship| {
                    ship.factoryinstancelist
                        .iter()
                        .map(|factory| {
                            factory
                                .outputs
                                .iter()
                                .filter(|output| {
                                    (output.resourcetype == self) & (output.propagate == true)
                                })
                                .map(|output| output.contents)
                                .sum::<u64>()
                        })
                        .sum::<u64>()
                })
                .sum::<u64>();
            let sum = (factorysupply + shipsupply) as f32;
            if sum == 0_f32 {
                None
            } else {
                Some(sum)
            }
        }
    }

    impl Salience<polarity::Demand> for ResourceID {
        const DEG_MULT: f32 = 1.0;
        fn get_value(
            self,
            (nodeid, node): (NodeID, &Node),
            shipinstances: &HashMap<ShipInstanceID, ShipInstance>,
            fleetinstances: &HashMap<FleetInstanceID, FleetInstance>,
        ) -> Option<f32> {
            let factorydemand: u64 = node
                .factoryinstancelist
                .iter()
                .map(|factory| {
                    factory
                        .inputs
                        .iter()
                        .filter(|input| (input.resourcetype == self) & (input.propagate == true))
                        .map(|input| input.target.saturating_sub(input.contents))
                        .sum::<u64>()
                })
                .sum::<u64>();
            let shipyarddemand: u64 = node
                .shipyardinstancelist
                .iter()
                .map(|shipyard| {
                    shipyard
                        .inputs
                        .iter()
                        .filter(|input| (input.resourcetype == self) & (input.propagate == true))
                        .map(|input| input.target.saturating_sub(input.contents))
                        .sum::<u64>()
                })
                .sum::<u64>();
            let shipdemand: u64 = shipinstances
                .values()
                .filter(|ship| ship.get_node(&shipinstances, &fleetinstances) == nodeid)
                .map(|ship| {
                    let factorydemand = ship
                        .factoryinstancelist
                        .iter()
                        .map(|factory| {
                            factory
                                .inputs
                                .iter()
                                .filter(|input| {
                                    (input.resourcetype == self) & (input.propagate == true)
                                })
                                .map(|input| input.target.saturating_sub(input.contents))
                                .sum::<u64>()
                        })
                        .sum::<u64>();
                    let shipyarddemand = ship
                        .shipyardinstancelist
                        .iter()
                        .map(|shipyard| {
                            shipyard
                                .inputs
                                .iter()
                                .filter(|input| {
                                    (input.resourcetype == self) & (input.propagate == true)
                                })
                                .map(|input| input.target.saturating_sub(input.contents))
                                .sum::<u64>()
                        })
                        .sum::<u64>();
                    factorydemand + shipyarddemand
                })
                .sum::<u64>();
            let sum = (factorydemand + shipyarddemand + shipdemand) as f32;
            if sum == 0_f32 {
                None
            } else {
                Some(sum)
            }
        }
    }

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct SystemID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
    pub struct NodeID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct NodeFlavorID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct FactionID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct ResourceID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct FactoryClassID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct FactoryInstanceID(pub usize);

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct ShipyardClassID(pub usize);

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct ShipyardInstanceID(pub usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct ShipClassID(pub usize);

    #[derive(Copy, Clone, Debug)]
    pub struct ShipAIID(usize);

    #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
    pub struct ShipInstanceID(usize);

    #[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
    pub struct FleetClassID(pub usize);

    #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
    pub struct FleetInstanceID(u64);

    #[derive(Copy, Clone, Debug)]
    pub struct ObjectiveID(usize);

    #[derive(Copy, Clone, Debug)]
    pub struct OperationID(usize);

    #[derive(Copy, Clone, Debug)]
    pub struct DeploymentID(usize);

    #[derive(Copy, Clone, Debug)]
    pub struct EngagementID(usize);

    #[derive(Debug, Hash, Clone, Eq, PartialEq)]
    pub struct System {
        pub visiblename: String,
        pub description: String,
        pub nodes: Vec<NodeID>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Node {
        pub visiblename: String, //location name as shown to player
        pub system: SystemID, //system in which node is located; this is used to generate all-to-all in-system links
        pub distance: u64, //node's distance from system star; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
        pub description: String,
        pub flavor: NodeFlavorID, //type of location this node is -- planet, asteroid field, hyperspace transit zone
        pub factoryinstancelist: Vec<FactoryInstance>, //this is populated at runtime from the factoryclasslist, not specified in json
        pub shipyardinstancelist: Vec<ShipyardInstance>,
        pub environment: String, //name of the FRED environment to use for missions set in this node
        pub allegiance: FactionID, //faction that currently holds the node
        pub efficiency: f64, //efficiency of any production facilities in this node; changes over time based on faction ownership
        pub threat: HashMap<FactionID, f32>,
    }

    impl Node {}

    #[derive(Debug, Hash, Clone, Eq, PartialEq)]
    pub struct NodeFlavor {
        pub visiblename: String,
        pub description: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Edges {
        hyperlinks: HashSet<(NodeID, NodeID)>, //list of links between nodes
        neighbormap: HashMap<NodeID, Vec<NodeID>>, //map of which nodes belong to which systems, for purposes of generating all-to-all links
    }

    impl Edges {
        fn insert(&mut self, a: NodeID, b: NodeID) {
            assert_ne!(a, b);
            self.hyperlinks.insert((a.max(b), a.min(b)));
            self.neighbormap
                .entry(a)
                .or_insert_with(|| Vec::new())
                .push(b);
            self.neighbormap
                .entry(b)
                .or_insert_with(|| Vec::new())
                .push(a);
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Faction {
        pub visiblename: String, //faction name as shown to player
        pub description: String,
        pub efficiencydefault: f64, //starting value for production facility efficiency
        pub efficiencytarget: f64, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
        pub efficiencydelta: f64,  //rate at which efficiency changes
        pub relations: HashMap<FactionID, f32>,
    }

    #[derive(Debug, Hash, Clone, Eq, PartialEq)]
    pub struct Resource {
        pub visiblename: String,
        pub description: String,
        pub cargovol: u64, //how much space one unit of this resource takes up when transported by a cargo ship
        pub valuemult: u64, //how valuable the AI considers one unit of this resource to be
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct Stockpile {
        pub resourcetype: ResourceID,
        pub contents: u64,
        pub rate: u64,
        pub target: u64,
        pub capacity: u64,
        pub propagate: bool,
    }

    impl Stockpile {
        fn input_is_sufficient(&self) -> bool {
            self.contents >= self.rate
        }

        //this is the logic to determine whether a FactoryOutputInstance should be active, dormant, or stalled
        fn output_state(&self) -> OutputState {
            if self.contents >= self.target {
                OutputState::Dormant
            } else if self.contents + self.rate >= self.capacity {
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
        fn output_process(&mut self, efficiency: f64) {
            self.contents += (self.rate as f64 * efficiency) as u64;
            if self.contents >= self.capacity {
                panic!("Output stockpile exceeds capacity.");
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct FactoryClass {
        pub visiblename: String,
        pub description: String,
        pub inputs: Vec<Stockpile>, //the data for the factory's asset consumption
        pub outputs: Vec<Stockpile>, //the data for the factory's asset production
    }

    impl FactoryClass {
        pub fn instantiate(&self, is_visible: bool) -> FactoryInstance {
            FactoryInstance {
                visiblename: self.visiblename.clone(),
                description: self.description.clone(),
                visibility: is_visible,
                inputs: self.inputs.clone(),
                outputs: self.outputs.clone(),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct FactoryInstance {
        //this is an actual factory, derived from a factory class
        visiblename: String,
        description: String,
        visibility: bool,
        inputs: Vec<Stockpile>,  //the data for the factory's asset consumption
        outputs: Vec<Stockpile>, //the data for the factory's asset production
    }

    //FactoryInstance sufficiency method
    impl FactoryInstance {
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
                    let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutputInstance sufficiency method defined earlier
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

        fn process(&mut self, location_efficiency: f64) {
            if let FactoryState::Active = self.get_state() {
                dbg!("Factory is active.");
                self.inputs
                    .iter_mut()
                    .for_each(|stockpile| stockpile.input_process());
                self.outputs
                    .iter_mut()
                    .for_each(|stockpile| stockpile.output_process(location_efficiency));
            } else {
                dbg!("Factory is inactive.");
            }
        }
    }

    enum FactoryState {
        Active,
        Dormant,
        Stalled,
    }

    enum OutputState {
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

    #[derive(Debug, Clone, PartialEq)]
    pub struct ShipyardClass {
        pub visiblename: Option<String>,
        pub description: Option<String>,
        pub inputs: Vec<Stockpile>,
        pub outputs: HashMap<ShipClassID, u64>,
        pub constructrate: u64,
        pub efficiency: f64,
    }

    impl ShipyardClass {
        pub fn instantiate(&self, is_visible: bool) -> ShipyardInstance {
            ShipyardInstance {
                visiblename: self.visiblename.clone(),
                description: self.description.clone(),
                visibility: is_visible,
                inputs: self.inputs.clone(),
                outputs: self.outputs.clone(),
                constructpoints: 0,
                constructrate: self.constructrate,
                efficiency: 1.0,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct ShipyardInstance {
        visiblename: Option<String>,
        description: Option<String>,
        visibility: bool,
        inputs: Vec<Stockpile>,
        outputs: HashMap<ShipClassID, u64>,
        constructpoints: u64,
        constructrate: u64,
        efficiency: f64,
    }

    impl ShipyardInstance {
        fn get_state(&self) -> FactoryState {
            let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutputInstance sufficiency method defined earlier
            if input_is_good {
                FactoryState::Active
            } else {
                FactoryState::Stalled //even if the factory's outputs are fine, we stall the factory if it doesn't have enough resources in its input stockpiles
            }
        }

        fn process(&mut self, location_efficiency: f64) {
            if let FactoryState::Active = self.get_state() {
                self.inputs
                    .iter_mut()
                    .for_each(|stockpile| stockpile.input_process());
                self.constructpoints += self.constructrate;
            }
        }

        fn try_choose_ship(&mut self, shipclasstable: &Vec<ShipClass>) -> Option<ShipClassID> {
            let shipclassid = self
                .outputs
                .iter()
                .max_by_key(|(_, weight)| *weight)
                .unwrap()
                .0;
            let cost = shipclasstable[shipclassid.0].basestrength;
            if self.constructpoints >= cost {
                self.constructpoints -= cost;
                Some(*shipclassid)
            } else {
                None
            }
        }

        fn plan_ships(
            &mut self,
            location_efficiency: f64,
            shipclasstable: &Vec<ShipClass>,
        ) -> Vec<ShipClassID> {
            self.process(location_efficiency);
            (0..)
                .map_while(|_| self.try_choose_ship(shipclasstable))
                .collect()
        }
    }
    #[derive(Debug, Clone)]
    pub struct ShipClass {
        pub id: ShipClassID,
        pub visiblename: String,
        pub description: String,
        pub basehull: Option<u64>, //how many hull hitpoints this ship has by default
        pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
        pub defaultweapons: Option<HashMap<ResourceID, u64>>, //a strikecraft's default weapons, which it always has with it
        pub hangarcap: Option<u64>, //this ship's capacity for carrying active strikecraft
        pub weaponcap: Option<u64>, //this ship's capacity for carrying strikecraft weapons
        pub cargocap: Option<u64>,  //this ship's capacity for carrying cargo
        pub hangarvol: Option<u64>, //how much hangar space this ship takes up when carried by a host
        pub cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
        pub factoryclasslist: Vec<FactoryClassID>,
        pub shipyardclasslist: Vec<ShipyardClassID>,
        pub hyperdrive: Option<u64>, //number of links this ship can traverse in one turn
        pub compconfig: Option<HashMap<ShipClassID, u64>>, //ideal configuration for this ship's strikecraft complement
        pub defectchance: HashMap<FactionID, f64>,         //
    }

    impl ShipClass {
        fn confighangarcap(&self, shipclasstable: &HashMap<ShipClassID, ShipClass>) -> Option<u64> {
            self.compconfig.as_ref().map(|cctbl| {
                cctbl
                    .iter()
                    .map(|(id, count)| shipclasstable.get(id).unwrap().hangarvol.unwrap() * count)
                    .sum()
            })
        }
        fn instantiate(
            &self,
            location: ShipLocationFlavor,
            faction: FactionID,
            factoryclasses: &Vec<FactoryClass>,
            shipyardclasses: &Vec<ShipyardClass>,
        ) -> ShipInstance {
            ShipInstance {
                visiblename: uuid::Uuid::new_v4().to_string(),
                shipclass: self.id,
                hull: self.basehull,
                strength: self.basestrength,
                factoryinstancelist: self
                    .factoryclasslist
                    .iter()
                    .map(|classid| factoryclasses[classid.0].instantiate(true))
                    .collect(),
                shipyardinstancelist: self
                    .shipyardclasslist
                    .iter()
                    .map(|classid| shipyardclasses[classid.0].instantiate(true))
                    .collect(),
                location,
                allegiance: faction,
                experience: 0,
                efficiency: 1.0,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct ShipAI {
        id: ShipAIID,
        ship_attract_specific: f64, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
        ship_attract_generic: f64, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
        resource_attract: HashMap<ResourceID, f64>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
    }

    #[derive(Debug)]
    pub struct ShipInstance {
        visiblename: String,
        shipclass: ShipClassID, //which class of ship this is
        hull: Option<u64>,      //how many hitpoints the ship has; strikecraft don't have hitpoints
        strength: u64, //ship's strength score, based on its class strength score but affected by its current hull percentage and experience score
        factoryinstancelist: Vec<FactoryInstance>,
        shipyardinstancelist: Vec<ShipyardInstance>,
        location: ShipLocationFlavor, //where the ship is -- a node if it's unaffiliated, a fleet if it's in one
        allegiance: FactionID,        //which faction this ship belongs to
        experience: u64, //XP gained by this ship, which affects strength score and in-mission AI class
        efficiency: f64,
    }

    impl ShipInstance {
        pub fn get_node(
            &self,
            shipinstances: &HashMap<ShipInstanceID, ShipInstance>,
            fleetinstances: &HashMap<FleetInstanceID, FleetInstance>,
        ) -> NodeID {
            match self.location {
                ShipLocationFlavor::Node(id) => id,
                ShipLocationFlavor::Fleet(id) => fleetinstances.get(&id).unwrap().location,
                ShipLocationFlavor::Host(flavor) => flavor.get_node(shipinstances, fleetinstances),
                ShipLocationFlavor::Detachment(node, fleet) => node,
            }
        }
    }

    #[derive(Debug, Copy, Clone)]
    enum ShipLocationFlavor {
        Node(NodeID),
        Fleet(FleetInstanceID),
        Host(HostFlavor),
        Detachment(NodeID, FleetInstanceID),
    }

    #[derive(Debug, Copy, Clone)]
    enum HostFlavor {
        Garrison(NodeID),
        Carrier(ShipInstanceID),
    }

    impl HostFlavor {
        pub fn get_node(
            &self,
            shipinstances: &HashMap<ShipInstanceID, ShipInstance>,
            fleetinstances: &HashMap<FleetInstanceID, FleetInstance>,
        ) -> NodeID {
            match self {
                HostFlavor::Garrison(id) => *id,
                HostFlavor::Carrier(id) => shipinstances
                    .get(id)
                    .unwrap()
                    .get_node(shipinstances, fleetinstances),
            }
        }
    }

    #[derive(Debug, Hash, Clone, Eq, PartialEq)]
    pub struct CargoStat {
        cargocap: u64,
        resourcecont: (ResourceID, u64),
        shipcont: Vec<ShipInstanceID>,
    }

    #[derive(Debug, Hash, Clone, Eq, PartialEq)]
    enum CargoFlavor {
        Resource((ResourceID, u64)),
        ShipInstance(Vec<ShipInstanceID>),
    }

    impl CargoFlavor {
        fn cargocapused(
            &self,
            resourcetable: &HashMap<ResourceID, Resource>,
            shipinstancetable: &HashMap<ShipInstanceID, ShipInstance>,
            shipclasstable: &HashMap<ShipClassID, ShipClass>,
        ) -> u64 {
            match self {
                Self::Resource((id, n)) => resourcetable.get(id).unwrap().cargovol * n,
                Self::ShipInstance(ids) => ids
                    .iter()
                    .map(|id| {
                        shipclasstable
                            .get(&shipinstancetable.get(id).unwrap().shipclass)
                            .unwrap()
                            .cargovol
                            .unwrap()
                    })
                    .sum(),
            }
        }
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct FleetClass {
        pub visiblename: String,
        pub description: String,
        pub fleetconfig: HashMap<ShipClassID, u64>,
    }

    #[derive(Debug)]
    pub struct FleetInstance {
        fleetclass: FleetClassID,
        location: NodeID,
        allegiance: FactionID,
    }

    #[derive(Debug)]
    pub struct Detachment {
        fleet: FleetInstanceID,
        location: NodeID,
        config: Vec<ShipInstanceID>,
        deployment: DeploymentID,
    }

    #[derive(Debug)]
    pub struct Objective {
        id: ObjectiveID,
        condition: ObjectiveFlavor, //?????????????? This is the condition for completing the objective but I have basically no idea how to do this. An enum of all the possible values? Or do I call a function here?
        cost: u64,
    }

    #[derive(Debug)]
    enum ObjectiveFlavor {
        ShipDeath {
            ship: ShipInstanceID,
        },
        ShipSafe {
            ship: ShipInstanceID,
            nturns: u64,
        },
        FleetDeath {
            fleet: FleetInstanceID,
        },
        FleetSafe {
            fleet: FleetInstanceID,
            nturns: u64,
            strengthpercentage: f64,
        },
        NodeCapture {
            node: NodeID,
        },
        NodeSafe {
            node: NodeID,
            nturns: u64,
        },
        SystemCapture {
            system: SystemID,
        },
        SystemSafe {
            system: SystemID,
            nturns: u64,
            nodespercentage: f64,
        },
    }

    #[derive(Debug)]
    pub struct Operation {
        id: OperationID,
        visiblename: String,
        fleet: FleetInstanceID,
        objectives: Vec<ObjectiveID>,
    }

    #[derive(Debug)]
    pub struct Deployment {
        id: DeploymentID,
        visiblename: String,
        fleet: FleetInstanceID,
        detachment: Vec<ShipInstanceID>,
        location: NodeID,
        objective: ObjectiveID,
    }

    #[derive(Debug)]
    pub struct Engagement {
        id: EngagementID,
        visiblename: String,
        fleets: HashMap<FactionID, Vec<FleetInstanceID>>,
        forces: HashMap<FactionID, Vec<ShipInstanceID>>,
        location: NodeID,
        objectives: HashMap<FactionID, ObjectiveID>,
    }
}

fn main() {
    let file = File::open("mod-specs.json").unwrap();
    let start = Instant::now();
    let json_root: json::Root = serde_json::from_reader(file).unwrap();
    let duration = start.elapsed();
    dbg!(duration);
    let mut root = json_root.hydrate();
    //for i in 0..50 {
    //    root.process_turn();
    //}
    //    dbg!(root.nodes);
    //    dbg!(root.shipinstances);
    //    dbg!(root.shipinstancecounter);
    let empire = internal::FactionID(0);
    let steel = internal::ResourceID(0);
    let components = internal::ResourceID(1);

    while root.turn < 5 {
        root.process_turn();
    }

    let threat_values = root
        .calculate_values::<internal::ResourceID, internal::polarity::Supply>(empire, empire, 5);
    salience_values
        .iter()
        .copied()
        .zip(root.nodes.iter().map(|node| node.visiblename.clone()))
        .for_each(|(value, node)| println!("{:.3}\t{}", value, node));

    let salience_values = root
        .calculate_values::<internal::ResourceID, internal::polarity::Supply>(components, empire, 5);
    salience_values
        .iter()
        .copied()
        .zip(root.nodes.iter().map(|node| node.visiblename.clone()))
        .for_each(|(value, node)| println!("{:.3}\t{}", value, node));



    root.process_turn();

    let salience_values = root
        .calculate_values::<internal::ResourceID, internal::polarity::Supply>(components, empire, 5);
    salience_values
        .iter()
        .copied()
        .zip(root.nodes.iter().map(|node| node.visiblename.clone()))
        .for_each(|(value, node)| println!("{:.3}\t{}", value, node));
}
