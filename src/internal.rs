use ordered_float::NotNan;
use std::cmp::Ordering;
use std::collections::{btree_map, hash_map, BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::sync::atomic::{self, AtomicI64};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct Key<T> {
    index: usize,
    phantom: PhantomData<T>,
}

impl<T> Ord for Key<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> PartialOrd for Key<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Key<T> {
    pub fn new_from_index(i: usize) -> Self {
        Key {
            index: i,
            phantom: PhantomData::default(),
        }
    }
}

impl<T> Hash for Key<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> PartialEq for Key<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<T> Eq for Key<T> {}

impl<T> Copy for Key<T> {}

impl<T> Clone for Key<T> {
    fn clone(&self) -> Key<T> {
        *self
    }
}

#[derive(Debug)]
pub struct Table<T> {
    next_index: usize,
    map: BTreeMap<Key<T>, T>,
}

impl<T> Table<T> {
    pub fn new() -> Self {
        Table {
            next_index: 0,
            map: BTreeMap::new(),
        }
    }
    pub fn put(&mut self, value: T) -> Key<T> {
        let key = Key {
            index: self.next_index,
            phantom: PhantomData::default(),
        };
        self.next_index += 1;
        self.map.insert(key, value);
        key
    }
    pub fn get(&self, key: Key<T>) -> &T {
        self.map
            .get(&key)
            .expect("Tried to get an entry that doesn't exist!")
    }
    pub fn get_mut(&mut self, key: Key<T>) -> &mut T {
        self.map
            .get_mut(&key)
            .expect("Tried to get an entry that doesn't exist!")
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn del(&mut self, key: Key<T>) {
        self.map
            .remove(&key)
            .expect("Tried to remove an entry that doesn't exist!");
    }
    pub fn iter(&self) -> btree_map::Iter<Key<T>, T> {
        self.map.iter()
    }
    pub fn iter_mut(&mut self) -> btree_map::IterMut<Key<T>, T> {
        self.map.iter_mut()
    }
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let mut table: Table<T> = Table::new();
        vec.into_iter().for_each(|entity| {
            table.put(entity);
        });
        table
    }
}

#[derive(Debug)]
pub struct Root {
    pub systems: Table<System>,
    pub nodes: Table<Node>,
    pub edges: HashSet<(Key<Node>, Key<Node>)>,
    pub neighbors: HashMap<Key<Node>, Vec<Key<Node>>>,
    pub factions: Table<Faction>,
    pub factoryclasses: Table<FactoryClass>,
    pub shipyardclasses: Table<ShipyardClass>,
    pub resources: Table<Resource>,
    pub shipais: Table<ShipAI>,
    pub shipclasses: Table<ShipClass>,
    pub shipinstances: Table<ShipInstance>,
    pub shipinstancecounter: usize,
    pub fleetclasses: Table<FleetClass>,
    pub fleetinstances: Table<FleetInstance>,
    pub turn: u64,
}

impl Root {
    pub fn process_turn(&mut self) {
        //we run the factory process for all factories attached to nodes, so that they produce and consume resources
        self.nodes.iter_mut().for_each(|(nodeid, node)| {
            node.factoryinstancelist
                .iter_mut()
                .for_each(|factory| factory.process(node.efficiency));
        });

        //here we create lists of ships all the shipyards attached to nodes should create
        let ship_plan_list: Vec<(Key<ShipClass>, ShipLocationFlavor, Key<Faction>)> = self
            .nodes
            .iter_mut()
            .map(|(&nodeid, node)| {
                node.shipyardinstancelist
                    .iter_mut()
                    .map(|shipyard| {
                        let ship_plans = shipyard.plan_ships(node.efficiency, &self.shipclasses);
                        //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                        ship_plans
                            .iter()
                            .map(|&ship_plan| {
                                (ship_plan, ShipLocationFlavor::Node(nodeid), node.allegiance)
                            })
                            // <^>>(
                            .collect::<Vec<_>>()
                    })
                    //we flatten the collection of vecs corresponding to individual shipyards, because we just want to create all the ships and don't care who created them
                    .flatten()
                    .collect::<Vec<_>>()
            })
            //we flatten a second time to get a node-agnostic list of all the ships we need to create
            .flatten()
            .collect();

        //this creates the planned ships, then reports how many ships were built this turn
        let n_newships = ship_plan_list
            .iter()
            .map(|&(id, location, faction)| self.create_ship(id, location, faction))
            .count();
        println!("Built {} new ships.", n_newships);

        for i in 0..10 {
            self.update_node_threats(10);
        }

        self.nodes.iter().for_each(|(_, node)| {
            let mut threat_list: Vec<(Key<Faction>, f32)> =
                node.threat.iter().map(|(fid, v)| (*fid, *v)).collect();
            threat_list.sort_by_key(|(id, _)| *id);
        });

        self.turn += 1;

        println!("It is now turn {}.", self.turn);
    }

    //this is the method for creating a ship
    //duh
    fn create_ship(
        &mut self,
        class_id: Key<ShipClass>,
        location: ShipLocationFlavor,
        faction: Key<Faction>,
    ) -> Key<ShipInstance> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        let new_ship = self.shipclasses.get(class_id).instantiate(
            location,
            faction,
            &self.factoryclasses,
            &self.shipyardclasses,
            &self.shipais,
        );
        self.shipinstancecounter += 1;
        //this will need to be changed when we switch ship instances to the table system
        //here we check to make sure the new ship's id doesn't already exist
        self.shipinstances.put(new_ship)
    }

    //we get the military strength of a node for a given faction by filtering down the global ship list by node and faction allegiance, then summing their strength values
    fn get_node_strength(&self, nodeid: Key<Node>, faction: Key<Faction>) -> u64 {
        self.shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(&self.shipinstances, &self.fleetinstances) == nodeid)
            .filter(|(_, ship)| ship.allegiance == faction)
            .map(|(_, ship)| ship.strength)
            .sum()
    }

    //oh god
    pub fn calculate_values<S: Salience<P> + Copy, P: Polarity>(
        //we need a salience, which is the type of resource or shipclass or whatever we're calculating values for
        //and the faction for which we're calculating values
        //and we specify the number of times we want to calculate these values, (NOTE: uncertain) i.e. the number of edges we'll propagate across
        &self,
        salience: S,
        subject_faction: Key<Faction>,
        n_iters: usize,
    ) -> Vec<f32> {
        //this map only contains the salience values being generated by things directly in each node, without any propagation
        //we filter down to only the nodes owned by the subject faction
        //then call get_value on the salience, and return the node id and salience value, while filtering down to only the nodes producing the subject salience
        //Length equals nodes owned by subject faction and producing subject salience
        let node_initial_salience_map: Vec<(Key<Node>, f32)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.allegiance == subject_faction)
            .filter_map(|(&id, node)| {
                salience
                    .get_value((id, node), &self.shipinstances, &self.fleetinstances)
                    .map(|v| (id, v))
            })
            .collect();
        //this map contains the amount of threat that exists from each faction, in each node, from the perspective of the subject faction
        //Length equals all nodes
        //This is a subjective map for subject faction
        let tagged_threats: Vec<HashMap<Key<Faction>, f32>> = self
            .nodes
            .iter()
            .map(|(id, node)| {
                //we iterate over the node's threat listing, and get the threat for each faction as perceived by the subject faction -- that is, multiplied by the subject faction's relations with that faction
                node.threat
                    .iter()
                    .map(|(f, t)| {
                        let value =
                            t * self.factions.get(subject_faction).relations.get(f).unwrap();
                        (*f, value)
                    })
                    .collect()
            })
            .collect();
        //this is the factor by which a salience passing through each node should be multiplied
        //we sum the tagged threats for each node -- which are valenced according to relations with the subject faction
        //then we use Alyssa's black mathemagics to convert them so that the scaling curve is correct
        //Length equals all nodes
        //This is a subjective map for subject faction
        let node_degradations: Vec<f32> = tagged_threats
            .iter()
            .map(|map| {
                let sum = map.values().sum();
                scale_from_threat(sum, 20_f32) * S::DEG_MULT * 0.8
            })
            .collect();
        //Outer vec length equals all nodes; inner vec equals nodes owned by faction and producing specified salience -- but only the inner node corresponding to the outer node has a nonzero value
        let node_salience_state: Vec<Vec<f32>> = self
            .nodes
            .iter()
            .map(|(id, node)| {
                //we iterate over the node initial salience map, which contains only nodes owned by subject faction and producing subject salience
                node_initial_salience_map
                    .iter()
                    //that gives us the initial salience value for each node
                    //we use this '== check as u8' to multiply it by 1 if the node matches the one the outer iterator is looking at, and multiply it by 0 otherwise
                    .map(|&(sourcenodeid, value)| value * ((sourcenodeid == *id) as u8) as f32)
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
            (0..n_iters).fold(node_salience_state, |mut state, n_iter| {
                println!("Completed {} iterations of salience propagation.", n_iter);
                self.edges.iter().for_each(|(a, b)| {
                    //we get the degradation scalar for each of the two nodes in the edge
                    let deg_a = node_degradations[a.index];
                    let deg_b = node_degradations[b.index];
                    //this loop does basically the same thing as an iterator but we have to do it this way for complicated ownership reasons
                    //we repeat the loop process n_tags times, 
                    for i in 0..n_tags {
                        //we index into node_salience_state's outer vec by node A's id, then into the inner vec by i; this means we're essentially iterating over the inner vec
                        //we update the i'th element of A (the inner vec) by taking the maximum between the i'th element of A and the i'th element of B, multiplied by node B's degradation scalar
                        //because this is the salience coming from node B to node A, getting degraded by B's threats as it leaves
                        state[a.index][i] = state[a.index][i].max(state[b.index][i] * deg_b);
                        //then we do the same thing again but backwards, to process the salience coming from node A to node B
                        state[b.index][i] = state[b.index][i].max(state[a.index][i] * deg_a);
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

    //this determines the threat for each faction present in each node, in absolute, objective terms
    //based on presence of military assets
    //n_steps determines how many edges the threat propagates across
    pub fn update_node_threats(&mut self, n_steps: usize) {
        let faction_threat: Vec<(Key<Faction>, Vec<f32>)> = self
            .factions
            .iter()
            .map(|(&id, _)| {
                let v = self.calculate_values(id, id, n_steps);
                (id, v)
            })
            .collect();
        faction_threat.iter().for_each(|(factionid, threat_list)| {
            threat_list
                .iter()
                .zip(self.nodes.iter_mut())
                .for_each(|(&threat_v, (_, node))| {
                    node.threat.insert(*factionid, threat_v).unwrap();
                })
        })
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

    pub struct Supply {}

    impl Polarity for Supply {}

    pub struct Demand {}

    impl Polarity for Demand {}
}

pub trait Salience<P: Polarity> {
    const DEG_MULT: f32;
    //this retrieves the value of a specific salience in a specific node
    fn get_value(
        self,
        node: (Key<Node>, &Node),
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
    ) -> Option<f32>;
}

//this method retrieves threat value generated by a given faction
impl Salience<polarity::Supply> for Key<Faction> {
    const DEG_MULT: f32 = 0.5;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
    ) -> Option<f32> {
        let node_strength: u64 = shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(shipinstances, fleetinstances) == nodeid)
            .filter(|(_, ship)| ship.allegiance == self)
            .map(|(_, ship)| ship.strength)
            .sum();
        Some(node_strength)
            .filter(|&strength| strength != 0)
            .map(|strength| strength as f32)
    }
}

//this method tells us how much supply there is of a given resource in a given node
impl Salience<polarity::Supply> for Key<Resource> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
    ) -> Option<f32> {
        //we add up all the resource quantity in factory output stockpiles in the node
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
        //then all the resource quantity in output stockpiles of factories attached to ships
        let shipsupply: u64 = shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(&shipinstances, &fleetinstances) == nodeid)
            .map(|(_, ship)| {
                ship.factoryinstancelist
                    .iter()
                    .map(|factory| {
                        factory
                            .outputs
                            .iter()
                            .filter(|output| {
                                //we only generate salience if the stockpile is set to propagate
                                //NOTE: I'm not 100% sure this is correct; should follow up
                                //I think it's fine though
                                (output.resourcetype == self) & (output.propagate == true)
                            })
                            .map(|output| output.contents)
                            .sum::<u64>()
                    })
                    .sum::<u64>()
            })
            .sum::<u64>();
        //then sum them together
        let sum = (factorysupply + shipsupply) as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

//this method tells us how much demand there is for a given resource in a given node
impl Salience<polarity::Demand> for Key<Resource> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
    ) -> Option<f32> {
        //add up resources from factory input stockpiles in node
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
        //add up resources from shipyard input stockpiles in node
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
        //now we have to look at ships in the node, since they might have stockpiles of their own
        let shipdemand: u64 = shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(&shipinstances, &fleetinstances) == nodeid)
            .map(|(_, ship)| {
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
        //and sum everything together
        let sum = (factorydemand + shipyarddemand + shipdemand) as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

//TODO: implement supply and demand for shipclasses, and make the logic apply more generally to stockpiles attached to ships

//most or all of these will be deprecated when we finish converting everything to the table system
#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
pub struct SystemID(pub usize);

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
pub struct NodeFlavorID(pub usize);

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
pub struct FactoryInstanceID(pub usize);

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ShipyardInstanceID(pub usize);

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
    pub nodes: Vec<Key<Node>>,
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
    pub allegiance: Key<Faction>, //faction that currently holds the node
    pub efficiency: f64, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub threat: HashMap<Key<Faction>, f32>,
}

impl Node {}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct NodeFlavor {
    pub visiblename: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edges {
    hyperlinks: HashSet<(Key<Node>, Key<Node>)>, //list of links between nodes
    neighbormap: HashMap<Key<Node>, Vec<Key<Node>>>, //map of which nodes belong to which systems, for purposes of generating all-to-all links
}

impl Edges {
    //this creates an edge between two nodes, and adds each node to the other's neighbor map
    fn insert(&mut self, a: Key<Node>, b: Key<Node>) {
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
    pub relations: HashMap<Key<Faction>, f32>,
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct Resource {
    pub visiblename: String,
    pub description: String,
    pub cargovol: u64, //how much space one unit of this resource takes up when transported by a cargo ship
    pub valuemult: u64, //how valuable the AI considers one unit of this resource to be
}

//GenericCargo is an entity which can be either a quantity of a resource or a shipinstance
#[derive(Debug, Clone, Copy)]
pub enum GenericCargo {
    Resource { id: Key<Resource>, value: u64 },
    ShipInstance(Key<ShipInstance>),
}

//CollatedCargo tells us the type of resource or ship that the cargo entity is; it's used in a hashmap with an integer denoting the quantity
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum CollatedCargo {
    Resource(Key<Resource>),
    ShipClass(Key<ShipClass>),
}

trait Stockpileness {
    fn get_contents(&self) -> Vec<GenericCargo>;
    fn collate_contents(&self) -> HashMap<CollatedCargo, u64>;
    fn get_capacity(&self) -> u64;
    fn get_allowed(&self) -> (Vec<Key<Resource>>, Vec<Key<ShipClass>>);
    fn insert_cargo(&mut self, cargo: GenericCargo);
    fn remove_cargo(&mut self, cargo: GenericCargo);
}

//this is a horrible incomprehensible nightmare that Amaryllis put me through for some reason
impl<A: Stockpileness, B: Stockpileness> Stockpileness for (A, B) {
    fn get_contents(&self) -> Vec<GenericCargo> {
        self.0
            .get_contents()
            .iter()
            .chain(self.1.get_contents().iter())
            .copied()
            .collect()
    }
    //It actually works now
    fn collate_contents(&self) -> HashMap<CollatedCargo, u64> {
        self.0
            .collate_contents()
            .iter()
            .chain(self.1.collate_contents().iter())
            .fold(HashMap::new(), |mut acc, (k, v)| {
                *acc.entry(*k).or_insert(0) += v;
                acc
            })
    }
    fn get_capacity(&self) -> u64 {
        self.0.get_capacity() + self.1.get_capacity()
    }
    fn get_allowed(&self) -> (Vec<Key<Resource>>, Vec<Key<ShipClass>>) {
        //self.0
        //    .collate_contents()
        //    .iter()
        //    .chain(self.1.collate_contents().iter())
        //    .collect()
        (Vec::new(), Vec::new())
    }
    fn insert_cargo(&mut self, cargo: GenericCargo) {}
    fn remove_cargo(&mut self, cargo: GenericCargo) {}
}

//a unipotent resource stockpile can contain only one type of resource, and it cannot contain ship instances
//however, the quantity of resource specified in the rate field may be added to or removed from the stockpile every turn, depending on how it's used
#[derive(Debug, Clone, PartialEq)]
pub struct UnipotentResourceStockpile {
    pub resourcetype: Key<Resource>,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for UnipotentResourceStockpile {
    fn get_contents(&self) -> Vec<GenericCargo> {
        vec![GenericCargo::Resource {
            id: self.resourcetype,
            value: self.contents,
        }]
    }
    fn collate_contents(&self) -> HashMap<CollatedCargo, u64> {
        iter::once((CollatedCargo::Resource(self.resourcetype), self.contents)).collect()
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_allowed(&self) -> (Vec<Key<Resource>>, Vec<Key<ShipClass>>) {
        (vec![self.resourcetype], Vec::new())
    }
    fn insert_cargo(&mut self, cargo: GenericCargo) {
        match cargo {
            GenericCargo::Resource { id, value } => {
                if id == self.resourcetype {
                    self.contents += value;
                } else {
                    panic!("Attempted to insert invalid resource!");
                }
            }
            _ => panic!("Non-resource objects cannot be inserted into a unipotent stockpile."),
        }
    }
    fn remove_cargo(&mut self, cargo: GenericCargo) {
        match cargo {
            GenericCargo::Resource { id, value } => {
                if id == self.resourcetype {
                    self.contents -= value;
                } else {
                    panic!("Attempted to remove invalid resource!");
                }
            }
            _ => panic!("Non-resource objects cannot be removed from a unipotent stockpile."),
        }
    }
}

impl UnipotentResourceStockpile {
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

//a pluripotent stockpile can contain any number of different resources and ships
//however, it has no constant rate of increase or decrease; things may only be added or removed manually
#[derive(Debug, Clone)]
pub struct PluripotentStockpile {
    pub contents: Vec<GenericCargo>,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for PluripotentStockpile {
    fn get_contents(&self) -> Vec<GenericCargo> {
        self.contents.clone()
    }
    fn collate_contents(&self) -> HashMap<CollatedCargo, u64> {
        self.contents.iter().fold(HashMap::new(), |mut acc, cargo| {
            //match cargo {
            //    GenericCargo::Resource {r_id, r_v} => {
            //        acc.insert(r_id, r_v)
            //    }
            //    GenericCargo::ShipInstance {s_id} => {
            //        acc.entry(s_id.shipclass).or_insert()
            //    }
            //}
            acc
        })
    }
    fn get_capacity(&self) -> u64 {
        0_u64
    }
    fn get_allowed(&self) -> (Vec<Key<Resource>>, Vec<Key<ShipClass>>) {
        (Vec::new(), Vec::new())
    }
    fn insert_cargo(&mut self, cargo: GenericCargo) {}
    fn remove_cargo(&mut self, cargo: GenericCargo) {}
}

//a given shared stockpile type has its contents shared between all instances of itself; it does not produce any salience propagation
//as it stands, it's possible for the resource used to go negative; we will probably want to go fix this later
//also we need to make methods
#[derive(Debug, Clone)]
pub struct SharedStockpile {
    pub resourcetype: Key<Resource>,
    pub contents: Arc<AtomicI64>,
    pub rate: u64,
    pub capacity: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactoryClass {
    pub visiblename: String,
    pub description: String,
    pub inputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset production
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
    inputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset consumption
    outputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset production
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

    //we take an active factory and update all its inputs and outputs to add or remove resources
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
    pub inputs: Vec<UnipotentResourceStockpile>,
    pub outputs: HashMap<Key<ShipClass>, u64>,
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
    inputs: Vec<UnipotentResourceStockpile>,
    outputs: HashMap<Key<ShipClass>, u64>,
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
            //NOTE: We need to multiply constructrate here by location_efficiency, but then we have to deal with the f64s and u64s interacting
            //and that's not actually all that hard but I don't have the energy for it right now
            self.constructpoints += self.constructrate;
        }
    }

    fn try_choose_ship(&mut self, shipclasstable: &Table<ShipClass>) -> Option<Key<ShipClass>> {
        //we go through the list of ships the shipyard can produce, specified as its outputs, and grab the one with the highest desirability weight
        let shipclassid = self
            .outputs
            .iter()
            .max_by_key(|(_, weight)| *weight)
            .unwrap()
            .0;
        let cost = shipclasstable.get(*shipclassid).basestrength;
        //then, if the shipyard has enough points to build it, we subtract the cost and return the ship class id
        if self.constructpoints >= cost {
            self.constructpoints -= cost;
            Some(*shipclassid)
        //otherwise we return nothing
        } else {
            None
        }
    }

    //this uses try_choose_ship to generate the list of ships the shipyard is building this turn
    fn plan_ships(
        &mut self,
        location_efficiency: f64,
        shipclasstable: &Table<ShipClass>,
    ) -> Vec<Key<ShipClass>> {
        self.process(location_efficiency);
        (0..)
            .map_while(|_| self.try_choose_ship(shipclasstable))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ShipAI {
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<Key<ShipClass>, f32>,
    pub resource_attract: HashMap<Key<Resource>, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
}

#[derive(Debug, Clone)]
pub struct ShipClass {
    pub id: Key<ShipClass>,
    pub visiblename: String,
    pub description: String,
    pub basehull: Option<u64>, //how many hull hitpoints this ship has by default
    pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub aiclass: Key<ShipAI>,
    pub defaultweapons: Option<HashMap<Key<Resource>, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangarcap: Option<u64>, //this ship's capacity for carrying active strikecraft
    pub weaponcap: Option<u64>, //this ship's capacity for carrying strikecraft weapons
    pub cargocap: Option<u64>,  //this ship's capacity for carrying cargo
    pub hangarvol: Option<u64>, //how much hangar space this ship takes up when carried by a host
    pub cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
    pub factoryclasslist: Vec<Key<FactoryClass>>,
    pub shipyardclasslist: Vec<Key<ShipyardClass>>,
    pub stockpiles: Vec<UnipotentResourceStockpile>,
    pub hyperdrive: Option<u64>, //number of links this ship can traverse in one turn
    pub compconfig: Option<HashMap<Key<ShipClass>, u64>>, //ideal configuration for this ship's strikecraft complement
    pub defectchance: HashMap<Key<Faction>, f64>,         //
}

impl ShipClass {
    //the hell is this?
    //okay I think I get it
    //this goes through the ship class's ideal configuration and adds up all the ships' hangar volume statistics to figure out how much space the whole ideal configuration takes up
    //I don't really understand why this is necessary given that the confighangarcap should usually be equal to the statted hangarcap
    //and if it's not I can't think of any reason off the top of my head why we'd need it?
    //but here it is I guess
    fn confighangarcap(&self, shipclasstable: &HashMap<Key<ShipClass>, ShipClass>) -> Option<u64> {
        self.compconfig.as_ref().map(|cctbl| {
            cctbl
                .iter()
                .map(|(id, count)| shipclasstable.get(id).unwrap().hangarvol.unwrap() * count)
                .sum()
        })
    }
    //method to create a ship instance with this ship class
    fn instantiate(
        &self,
        location: ShipLocationFlavor,
        faction: Key<Faction>,
        factoryclasses: &Table<FactoryClass>,
        shipyardclasses: &Table<ShipyardClass>,
        shipais: &Table<ShipAI>,
    ) -> ShipInstance {
        ShipInstance {
            visiblename: uuid::Uuid::new_v4().to_string(),
            shipclass: self.id,
            hull: self.basehull,
            strength: self.basestrength,
            factoryinstancelist: self
                .factoryclasslist
                .iter()
                .map(|classid| factoryclasses.get(*classid).instantiate(true))
                .collect(),
            shipyardinstancelist: self
                .shipyardclasslist
                .iter()
                .map(|classid| shipyardclasses.get(*classid).instantiate(true))
                .collect(),
            stockpiles: self.stockpiles.clone(),
            location,
            allegiance: faction,
            experience: 0,
            efficiency: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct ShipInstance {
    visiblename: String,
    shipclass: Key<ShipClass>, //which class of ship this is
    hull: Option<u64>,         //how many hitpoints the ship has; strikecraft don't have hitpoints
    strength: u64, //ship's strength score, based on its class strength score but affected by its current hull percentage and experience score
    factoryinstancelist: Vec<FactoryInstance>,
    shipyardinstancelist: Vec<ShipyardInstance>,
    stockpiles: Vec<UnipotentResourceStockpile>,
    location: ShipLocationFlavor, //where the ship is -- a node if it's unaffiliated, a fleet if it's in one
    allegiance: Key<Faction>,     //which faction this ship belongs to
    experience: u64, //XP gained by this ship, which affects strength score and in-mission AI class
    efficiency: f64,
}

impl ShipInstance {
    //determines which node the ship is in
    //a ship can be in a number of places which aren't directly in a node, but all of them cash out to a node eventually
    pub fn get_node(
        &self,
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
    ) -> Key<Node> {
        match self.location {
            ShipLocationFlavor::Node(id) => id,
            ShipLocationFlavor::Fleet(id) => fleetinstances.get(id).location,
            ShipLocationFlavor::Host(flavor) => flavor.get_node(shipinstances, fleetinstances),
            ShipLocationFlavor::Detachment(node, fleet) => node,
        }
    }
    pub fn navigate(
        //used for ships which are operating independently
        //this method determines which of the current nodes neighbors is most desirable
        &self,
        nodes: &Vec<Node>,
        neighbors: &HashMap<Key<Node>, Vec<Key<Node>>>,
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
        //NOTE: I have flipped around the order of the vecs here in the resource and shipclass salience maps
        //because that will make it easier to get the necessary map out of calculate_values
        resource_salience_map: &Vec<Vec<f32>>, //outer vec is resources, inner vec is nodes
        shipclass_salience_map: &Vec<Vec<f32>>, //outer vec is shipclasses, inner vec is nodes
        shipclasses: &Table<ShipClass>,
        shipais: &Table<ShipAI>,
    ) -> Key<Node> {
        let position: Key<Node> = self.get_node(&shipinstances, &fleetinstances);
        //the AI of the ship we're looking at
        let self_ai = shipclasses.get(self.shipclass).aiclass;
        //we get the neighbor map of the ship's position, then iterate over it to determine which neighbor is most desirable
        *neighbors
            .get(&position)
            .unwrap()
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .max_by_key(|nodeid| {
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                let resource_value: f32 = shipais
                    .get(self_ai)
                    .resource_attract
                    .iter()
                    .map(|(resourceid, scalar)| {
                        //NOTE: This line and others using the salience maps used to have the indexes in the opposite order.
                        //I have switched them around to better fit with what calculate_values gives us.
                        //Possibly this will break something somehow.
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        resource_salience_map[resourceid.index][nodeid.index] * scalar
                    })
                    .sum();
                //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
                let ship_cargo_value: f32 = shipais
                    .get(self_ai)
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        //we index into the salience map by shipclass and then by node
                        //to determine how much supply there is in this node for each shipclass the subject ship wants
                        shipclass_salience_map[shipclassid.index][nodeid.index] * scalar
                    })
                    .sum();
                //this checks how much demand there is in the node for ships of the subject ship's class
                let ship_value_specific: f32 = shipclass_salience_map[self.shipclass.index]
                    [nodeid.index]
                    * shipais.get(self_ai).ship_attract_specific;
                //oh, THIS is why we needed the placeholder ship class
                //this checks how much demand there is in the node for ships in general
                let ship_value_generic: f32 = shipclass_salience_map[0][nodeid.index]
                    * shipais.get(self_ai).ship_attract_generic;

                NotNan::new(resource_value + ship_value_specific + ship_value_generic).unwrap()
            })
            //if this doesn't work for some reason, like if the current node has no neighbors, the ship just stays where it is
            .unwrap_or(&position)
    }
}

#[derive(Debug, Copy, Clone)]
enum ShipLocationFlavor {
    Node(Key<Node>),
    Fleet(Key<FleetInstance>),
    Host(HostFlavor),
    Detachment(Key<Node>, Key<FleetInstance>),
}

#[derive(Debug, Copy, Clone)]
enum HostFlavor {
    Garrison(Key<Node>),
    Carrier(Key<ShipInstance>),
}

impl HostFlavor {
    pub fn get_node(
        &self,
        shipinstances: &Table<ShipInstance>,
        fleetinstances: &Table<FleetInstance>,
    ) -> Key<Node> {
        match self {
            HostFlavor::Garrison(id) => *id,
            HostFlavor::Carrier(id) => shipinstances
                .get(*id)
                .get_node(shipinstances, fleetinstances),
        }
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct CargoStat {
    cargocap: u64,
    resourcecont: (Key<Resource>, u64),
    shipcont: Vec<Key<ShipInstance>>,
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
enum CargoFlavor {
    Resource((Key<Resource>, u64)),
    ShipInstance(Vec<Key<ShipInstance>>),
}

impl CargoFlavor {
    fn cargocapused(
        &self,
        resourcetable: &HashMap<Key<Resource>, Resource>,
        shipinstancetable: &Table<ShipInstance>,
        shipclasstable: &HashMap<Key<ShipClass>, ShipClass>,
    ) -> u64 {
        match self {
            Self::Resource((id, n)) => resourcetable.get(id).unwrap().cargovol * n,
            Self::ShipInstance(ids) => ids
                .iter()
                .map(|&id| {
                    shipclasstable
                        .get(&shipinstancetable.get(id).shipclass)
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
    pub fleetconfig: HashMap<Key<ShipClass>, u64>,
}

#[derive(Debug)]
pub struct FleetInstance {
    fleetclass: Key<FleetClass>,
    location: Key<Node>,
    allegiance: Key<Faction>,
}

#[derive(Debug)]
pub struct Detachment {
    fleet: Key<FleetInstance>,
    location: Key<Node>,
    config: Vec<Key<ShipInstance>>,
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
        ship: Key<ShipInstance>,
    },
    ShipSafe {
        ship: Key<ShipInstance>,
        nturns: u64,
    },
    FleetDeath {
        fleet: Key<FleetInstance>,
    },
    FleetSafe {
        fleet: Key<FleetInstance>,
        nturns: u64,
        strengthpercentage: f64,
    },
    NodeCapture {
        node: Key<Node>,
    },
    NodeSafe {
        node: Key<Node>,
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
    fleet: Key<FleetInstance>,
    objectives: Vec<ObjectiveID>,
}

#[derive(Debug)]
pub struct Deployment {
    id: DeploymentID,
    visiblename: String,
    fleet: Key<FleetInstance>,
    detachment: Vec<Key<ShipInstance>>,
    location: Key<Node>,
    objective: ObjectiveID,
}

#[derive(Debug)]
pub struct Engagement {
    id: EngagementID,
    visiblename: String,
    fleets: HashMap<Key<Faction>, Vec<Key<FleetInstance>>>,
    forces: HashMap<Key<Faction>, Vec<Key<ShipInstance>>>,
    location: Key<Node>,
    objectives: HashMap<Key<Faction>, ObjectiveID>,
}
