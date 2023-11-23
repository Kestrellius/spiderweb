use average::Mean;
use no_panic::no_panic;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_distr::*;
use std::cmp::Ordering;
use std::collections::{btree_map, hash_map, BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::sync::atomic::{self, AtomicU64};
use std::sync::{Arc, Mutex};

pub struct Key<T> {
    index: usize,
    phantom: PhantomData<T>,
}

impl<T> fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key: {}", self.index)
    }
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
    pub fn get(&self, key: Key<T>) -> Option<&T> {
        self.map.get(&key)
    }
    pub fn get_mut(&mut self, key: Key<T>) -> Option<&mut T> {
        self.map.get_mut(&key)
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
    pub fn retain<F: FnMut(&Key<T>, &mut T) -> bool>(&mut self, f: F) {
        self.map.retain(f);
    }
    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut table: Table<T> = Table::new();
        vec.into_iter().for_each(|entity| {
            table.put(entity);
        });
        table
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct NodeFlavor {
    pub visiblename: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub visiblename: String, //location name as shown to player
    pub system: Key<System>, //system in which node is located; this is used to generate all-to-all in-system links
    pub position: [i64; 3], //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    pub description: String,
    pub flavor: Key<NodeFlavor>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub factoryinstancelist: Vec<FactoryInstance>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub environment: String, //name of the FRED environment to use for missions set in this node
    pub bitmap: Option<(String, f32)>, //name of the bitmap used to depict this node in other nodes' FRED environments, and a size factor
    pub allegiance: Key<Faction>,      //faction that currently holds the node
    pub efficiency: f64, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub threat: HashMap<Key<Faction>, f32>,
}

impl Node {
    pub fn get_node_forces(
        node: Key<Node>,
        root: &Root,
    ) -> HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)> {
        root.factions
            .iter()
            .map(|(factionid, _)| {
                let ships: Vec<Key<ShipInstance>> = root
                    .shipinstances
                    .iter()
                    .filter(|(_, ship)| ship.allegiance == *factionid)
                    .filter(|(_, ship)| ship.get_node(root) == node)
                    .map(|(shipid, _)| *shipid)
                    .collect();
                let fleets: Vec<Key<FleetInstance>> = root
                    .fleetinstances
                    .iter()
                    .filter(|(_, fleet)| fleet.allegiance == *factionid)
                    .filter(|(_, fleet)| fleet.location == node)
                    .map(|(fleetid, _)| *fleetid)
                    .collect();
                (*factionid, (fleets, ships))
            })
            .filter(|(_, (_, ships))| ships.len() > 0)
            .collect()
    }
    pub fn get_node_faction_reinforcements(
        location: Key<Node>,
        destination: Key<Node>,
        factionid: Key<Faction>,
        root: &Root,
    ) -> (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>) {
        let relevant_ships: HashMap<&Key<ShipInstance>, &ShipInstance> = root
            .shipinstances
            .iter()
            .filter(|(_, ship)| ship.allegiance == factionid)
            .filter(|(_, ship)| ship.get_node(root) == location)
            .collect();
        let (active_fleets, passive_fleets): (Vec<_>, Vec<_>) = root
            .fleetinstances
            .iter()
            .filter(|(_, fleet)| fleet.allegiance == factionid)
            .filter(|(_, fleet)| fleet.location == location)
            .map(|(id, fleet)| *id)
            .partition(|id| {
                //NOTE: this is fucky and I'd love for it to be less fucky
                root.fleetinstances
                    .get(*id)
                    .unwrap()
                    .nav_check(location, destination)
            });
        let ships_in_active_fleets: Vec<Key<ShipInstance>> = relevant_ships
            .iter()
            .filter(|(_, ship)| {
                active_fleets.contains(
                    &ship
                        .get_fleet(root)
                        .unwrap_or(Key::new_from_index(usize::MAX)),
                )
            })
            .map(|(shipid, ship)| {
                let mut vec = ship.get_daughters(&root.shipinstances);
                vec.insert(0, **shipid);
                vec
            })
            .flatten()
            .collect();
        let mut ships_in_node: Vec<Key<ShipInstance>> = relevant_ships
            .iter()
            .filter(|(_, ship)| ship.is_in_node(root))
            .filter(|(_, ship)| ship.nav_check(root, destination))
            .map(|(shipid, ship)| {
                let mut vec = ship.get_daughters(&root.shipinstances);
                vec.insert(0, **shipid);
                vec
            })
            .flatten()
            .collect();
        let mut passive_ships: Vec<Key<ShipInstance>> = relevant_ships
            .iter()
            .filter(|(id, ship)| {
                (ship.is_in_node(root) && !ships_in_node.contains(*id))
                    || passive_fleets.contains(
                        &ship
                            .get_fleet(root)
                            .unwrap_or(Key::new_from_index(usize::MAX)),
                    )
            })
            .map(|(shipid, ship)| {
                let mut vec = ship.get_checked_daughters(root, destination);
                vec.insert(0, **shipid);
                vec
            })
            .flatten()
            .collect();
        let mut ships: Vec<Key<ShipInstance>> = ships_in_active_fleets;
        ships.append(&mut ships_in_node);
        ships.append(&mut passive_ships);
        (active_fleets, ships)
    }
    pub fn get_node_faction_forces(
        node: Key<Node>,
        factionid: Key<Faction>,
        root: &Root,
    ) -> (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>) {
        let ships: Vec<Key<ShipInstance>> = root
            .shipinstances
            .iter()
            .filter(|(_, ship)| ship.allegiance == factionid)
            .filter(|(_, ship)| ship.get_node(root) == node)
            .map(|(shipid, _)| *shipid)
            .collect();
        let fleets: Vec<Key<FleetInstance>> = root
            .fleetinstances
            .iter()
            .filter(|(_, fleet)| fleet.allegiance == factionid)
            .filter(|(_, fleet)| fleet.location == node)
            .map(|(fleetid, _)| *fleetid)
            .collect();
        (fleets, ships)
    }
    pub fn get_system(node: Key<Node>, root: &Root) -> Option<Key<System>> {
        let system = root.systems.iter().find(|(_, s)| s.nodes.contains(&node));
        match system {
            Some((id, _)) => Some(*id),
            None => None,
        }
    }
    pub fn is_in_system(node: Key<Node>, system: Key<System>, root: &Root) -> bool {
        root.systems.get(system).unwrap().nodes.contains(&node)
    }
    pub fn get_distance(a: Key<Node>, b: Key<Node>, root: &Root) -> u64 {
        let a_pos = root.nodes.get(a).unwrap().position;
        let b_pos = root.nodes.get(b).unwrap().position;
        (((a_pos[0] - b_pos[0]) + (a_pos[1] - b_pos[1]) + (a_pos[2] - b_pos[2])) as f64).sqrt()
            as u64
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct System {
    pub visiblename: String,
    pub description: String,
    pub nodes: Vec<Key<Node>>,
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct EdgeFlavor {
    pub visiblename: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edges {
    hyperlinks: HashSet<(Key<Node>, Key<Node>, Key<EdgeFlavor>)>, //list of links between nodes
    neighbormap: HashMap<Key<Node>, Vec<Key<Node>>>, //NOTE: investigate. Map of which nodes belong to which systems, for purposes of generating all-to-all links
}

impl Edges {
    //this creates an edge between two nodes, and adds each node to the other's neighbor map
    fn insert(&mut self, a: Key<Node>, b: Key<Node>, flavor: Key<EdgeFlavor>) {
        assert_ne!(a, b);
        self.hyperlinks.insert((a.max(b), a.min(b), flavor));
        self.neighbormap
            .entry(a)
            .or_insert_with(|| Vec::new())
            .push(b);
        self.neighbormap
            .entry(b)
            .or_insert_with(|| Vec::new())
            .push(a);
    }
    /*fn insert_with_distance(&mut self, root: &mut Root, a: Key<Node>, b: Key<Node>, distance: u64) {
        for i in 0..=distance {
            let p = root.create_node(0, None, None, null, etc);
            self.insert(a, p)
        }
    }*/
}

#[derive(Debug, Clone, PartialEq)]
pub struct Faction {
    pub visiblename: String, //faction name as shown to player
    pub description: String,
    pub efficiencydefault: f64, //starting value for production facility efficiency
    pub efficiencytarget: f64, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    pub efficiencydelta: f64,  //rate at which efficiency changes
    pub battlescalar: f32,
    pub relations: HashMap<Key<Faction>, f32>,
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct Resource {
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub cargovol: u64, //how much space one unit of this resource takes up when transported by a cargo ship
    pub valuemult: u64, //how valuable the AI considers one unit of this resource to be
}

//GenericCargo is an entity which can be either a quantity of a resource or a shipinstance
#[derive(Debug, Clone, Copy)]
pub enum GenericCargo {
    Resource { id: Key<Resource>, value: u64 },
    ShipInstance(Key<ShipInstance>),
}

impl GenericCargo {
    fn is_resource(self) -> Option<(Key<Resource>, u64)> {
        match self {
            GenericCargo::Resource { id, value } => Some((id, value)),
            _ => None,
        }
    }
    fn is_shipinstance(self) -> Option<Key<ShipInstance>> {
        match self {
            GenericCargo::ShipInstance(k) => Some(k),
            _ => None,
        }
    }
}

//CollatedCargo tells us the type of resource or ship that the cargo entity is; it's used in a hashmap with an integer denoting the quantity
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum CollatedCargo {
    Resource(Key<Resource>),
    ShipClass(Key<ShipClass>),
}

impl CollatedCargo {
    fn get_volume(self, root: &Root) -> u64 {
        match self {
            CollatedCargo::Resource(k) => root.resources.get(k).unwrap().cargovol,
            CollatedCargo::ShipClass(k) => root
                .shipclasses
                .get(k)
                .unwrap()
                .cargovol
                .unwrap_or(u64::MAX),
        }
    }
}

pub trait Stockpileness {
    fn get_resource_contents(&self) -> HashMap<Key<Resource>, u64>;
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>>;
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64>;
    fn get_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64;
    fn get_shipclass_num(&self, root: &Root, cargo: Key<ShipClass>) -> u64;
    fn get_capacity(&self) -> u64;
    fn get_fullness(&self, root: &Root) -> u64;
    fn get_allowed(&self) -> Option<(Vec<Key<Resource>>, Vec<Key<ShipClass>>)>;
    fn get_resource_supply(&self, root: &Root, resourceid: Key<Resource>) -> u64;
    fn get_resource_demand(&self, root: &Root, resourceid: Key<Resource>) -> u64;
    fn get_shipclass_supply(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64;
    fn get_shipclass_demand(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64;
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo>;
    fn remove(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo>;
    fn transfer<S: Stockpileness>(
        &mut self,
        rhs: &mut S,
        root: &Root,
        class: CollatedCargo,
        quantity: u64,
    ) {
        let available_space = (rhs.get_capacity() - rhs.get_fullness(root))
            .checked_div(class.get_volume(root))
            .unwrap_or(u64::MAX);
        let constrained_quantity = std::cmp::min(quantity, available_space);
        let cargo: Vec<GenericCargo> = match class {
            CollatedCargo::Resource(k) => vec![GenericCargo::Resource {
                id: k,
                value: constrained_quantity,
            }],
            CollatedCargo::ShipClass(k) => self
                .get_ship_contents()
                .iter()
                .filter(|ship| root.shipinstances.get(**ship).unwrap().shipclass == k)
                .take(constrained_quantity as usize)
                .map(|ship| GenericCargo::ShipInstance(*ship))
                .collect(),
        };
        dbg!(&cargo);
        cargo.iter().for_each(|item| {
            self.remove(root, *item)
                .and_then(|remainder| rhs.insert(root, remainder))
                .and_then(|remainder| self.insert(root, remainder));
        })
    }
}

//this is a horrible incomprehensible nightmare that Amaryllis put me through for some reason
//okay, so, a year later, what this actually does is that it takes two individual stockpiles and allows them to function together as a single stockpile
impl<A: Stockpileness, B: Stockpileness> Stockpileness for (A, B) {
    fn get_resource_contents(&self) -> HashMap<Key<Resource>, u64> {
        self.0
            .get_resource_contents()
            .iter()
            .chain(self.1.get_resource_contents().iter())
            .map(|(&k, &v)| (k, v))
            .collect()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        self.0
            .get_ship_contents()
            .iter()
            .chain(self.1.get_ship_contents().iter())
            .map(|&k| k)
            .collect()
    }
    //It actually works now
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        self.0
            .collate_contents(root)
            .iter()
            .chain(self.1.collate_contents(root).iter())
            .fold(HashMap::new(), |mut acc, (k, v)| {
                *acc.entry(*k).or_insert(0) += v;
                acc
            })
    }
    fn get_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        self.0.get_resource_num(root, cargo) + self.1.get_resource_num(root, cargo)
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        self.0.get_shipclass_num(root, cargo) + self.1.get_shipclass_num(root, cargo)
    }
    fn get_capacity(&self) -> u64 {
        self.0.get_capacity() + self.1.get_capacity()
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.0.get_fullness(root) + self.1.get_fullness(root)
    }
    fn get_allowed(&self) -> Option<(Vec<Key<Resource>>, Vec<Key<ShipClass>>)> {
        //self.0
        //    .get_allowed()
        //    .iter()
        //    .chain(self.1.get_allowed().iter())
        //    .collect()
        Some((Vec::new(), Vec::new()))
    }
    fn get_resource_supply(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.0.get_resource_supply(root, resourceid) + self.1.get_resource_supply(root, resourceid)
    }
    fn get_resource_demand(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.0.get_resource_demand(root, resourceid) + self.1.get_resource_demand(root, resourceid)
    }
    fn get_shipclass_supply(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        self.0.get_shipclass_supply(root, shipclassid)
            + self.1.get_shipclass_supply(root, shipclassid)
    }
    fn get_shipclass_demand(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        self.0.get_shipclass_demand(root, shipclassid)
            + self.1.get_shipclass_demand(root, shipclassid)
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        None
    }
    fn remove(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        None
    }
}

//a unipotent resource stockpile can contain only one type of resource, and it cannot contain ship instances
//however, the quantity of resource specified in the rate field may be added to or removed from the stockpile under various circumstances,
//such as once every turn, depending on how it's used
#[derive(Debug, Clone, PartialEq)]
pub struct UnipotentResourceStockpile {
    pub visibility: bool,
    pub resourcetype: Key<Resource>,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for UnipotentResourceStockpile {
    fn get_resource_contents(&self) -> HashMap<Key<Resource>, u64> {
        iter::once((self.resourcetype, self.contents)).collect()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        HashSet::new()
    }
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        iter::once((CollatedCargo::Resource(self.resourcetype), self.contents)).collect()
    }
    fn get_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        if cargo == self.resourcetype {
            self.contents
        } else {
            0
        }
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        0
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.contents * root.resources.get(self.resourcetype).unwrap().cargovol
    }
    fn get_allowed(&self) -> Option<(Vec<Key<Resource>>, Vec<Key<ShipClass>>)> {
        Some((vec![self.resourcetype], Vec::new()))
    }
    fn get_resource_supply(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        if resourceid == self.resourcetype {
            (self.contents * root.resources.get(resourceid).unwrap().cargovol)
                .saturating_sub(self.target)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        if resourceid == self.resourcetype {
            self.target
                .saturating_sub(self.contents * root.resources.get(resourceid).unwrap().cargovol)
        } else {
            0
        }
    }
    fn get_shipclass_supply(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        0
    }
    fn get_shipclass_demand(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        0
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo {
            GenericCargo::Resource { id, value } => {
                let cargo_vol = root.resources.get(id).unwrap().cargovol;
                if id == self.resourcetype {
                    let old_contents = self.contents;
                    let count_capacity = self.capacity / cargo_vol;
                    let remainder = value.saturating_sub(count_capacity - self.contents);
                    self.contents += value - remainder;
                    println!("Inserting {}.", value - remainder);
                    assert!(self.contents <= count_capacity);
                    assert_eq!(self.contents + remainder, old_contents + value);
                    Some(GenericCargo::Resource {
                        id: id,
                        value: remainder,
                    })
                } else {
                    //this will just hand the cargo back to whoever was trying to put it in
                    println!("Wrong resource type! No insertion.");
                    Some(cargo)
                }
            }
            _ => {
                println!("Ship, not resource! No insertion.");
                Some(cargo)
            }
        }
    }
    fn remove(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo {
            GenericCargo::Resource { id, value } => {
                if id == self.resourcetype {
                    let remainder = value.saturating_sub(self.contents);
                    self.contents -= value - remainder;
                    println!("Removing {}.", value - remainder);
                    Some(GenericCargo::Resource {
                        id: id,
                        value: (value - remainder),
                    })
                } else {
                    //we can't get out something the stockpile can't hold
                    Some(GenericCargo::Resource { id: id, value: 0 })
                }
            }
            _ => None,
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
    pub visibility: bool,
    pub resource_contents: HashMap<Key<Resource>, u64>,
    pub ship_contents: HashSet<Key<ShipInstance>>,
    pub allowed: Option<(Vec<Key<Resource>>, Vec<Key<ShipClass>>)>,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for PluripotentStockpile {
    fn get_resource_contents(&self) -> HashMap<Key<Resource>, u64> {
        self.resource_contents.clone()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        self.ship_contents.clone()
    }
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        let resource_list = self
            .resource_contents
            .iter()
            .map(|(key, value)| (CollatedCargo::Resource(*key), *value));
        let ship_list = self.ship_contents.iter().map(|key| {
            (
                CollatedCargo::ShipClass(root.shipinstances.get(*key).unwrap().shipclass),
                1,
            )
        });
        resource_list
            .chain(ship_list)
            .fold(HashMap::new(), |mut acc, (cc, num)| {
                *acc.entry(cc).or_insert(0) += num;
                acc
            })
    }
    fn get_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        *self.resource_contents.get(&cargo).unwrap_or(&0)
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        self.ship_contents
            .iter()
            .filter(|key| root.shipinstances.get(**key).unwrap().shipclass == cargo)
            .fold(0, |mut acc, _| {
                acc += 1;
                acc
            })
    }
    fn get_capacity(&self) -> u64 {
        self.capacity.clone()
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.resource_contents
            .iter()
            .map(|(id, value)| value * root.resources.get(*id).unwrap().cargovol)
            .sum::<u64>()
            + self
                .ship_contents
                .iter()
                .map(|key| {
                    root.shipclasses
                        .get(root.shipinstances.get(*key).unwrap().shipclass)
                        .unwrap()
                        .cargovol
                        .unwrap_or(0)
                })
                .sum::<u64>()
    }
    fn get_allowed(&self) -> Option<(Vec<Key<Resource>>, Vec<Key<ShipClass>>)> {
        self.allowed.clone()
    }
    //unlike other places, here in pluripotent stockpiles we don't take target into account when calculating supply
    //thus, items in pluripotent stockpiles always emit supply, even if the stockpile still wants more
    fn get_resource_supply(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.get_resource_num(root, resourceid) * root.resources.get(resourceid).unwrap().cargovol
    }
    fn get_resource_demand(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        if self
            .get_allowed()
            .unwrap_or((vec![resourceid], Vec::new()))
            .0
            .contains(&resourceid)
        {
            self.target.saturating_sub(self.get_fullness(root))
        } else {
            0
        }
    }
    fn get_shipclass_supply(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        self.get_shipclass_num(root, shipclassid)
            * root
                .shipclasses
                .get(shipclassid)
                .unwrap()
                .cargovol
                .unwrap_or(0)
    }
    fn get_shipclass_demand(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        if self
            .get_allowed()
            .unwrap_or((Vec::new(), vec![shipclassid]))
            .1
            .contains(&shipclassid)
        {
            self.target.saturating_sub(self.get_fullness(root))
        } else {
            0
        }
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo {
            GenericCargo::Resource { id, value } => {
                if self
                    .allowed
                    .clone()
                    .map(|(x, _)| x.contains(&id))
                    .unwrap_or(true)
                {
                    let cargo_vol = root.resources.get(id).unwrap().cargovol;
                    let fullness = self.get_fullness(root);
                    let how_many_fit = (self.capacity - fullness) / cargo_vol;
                    let remainder = value.saturating_sub(how_many_fit);
                    *self.resource_contents.get_mut(&id).unwrap() += value - remainder;
                    Some(GenericCargo::Resource {
                        id: id,
                        value: remainder,
                    })
                } else {
                    Some(cargo)
                }
            }
            GenericCargo::ShipInstance(id) => {
                let classid = root.shipinstances.get(id).unwrap().shipclass;
                if self
                    .allowed
                    .clone()
                    .map(|(_, x)| x.contains(&classid))
                    .unwrap_or(true)
                {
                    let cargo_vol = root.shipclasses.get(classid).unwrap().cargovol.unwrap();
                    if cargo_vol <= (self.capacity - self.get_fullness(root)) {
                        self.ship_contents.insert(id);
                        None
                    } else {
                        Some(cargo)
                    }
                } else {
                    Some(cargo)
                }
            }
        }
    }
    fn remove(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo {
            GenericCargo::Resource { id, value } => {
                if self
                    .allowed
                    .clone()
                    .map(|(x, _)| x.contains(&id))
                    .unwrap_or(true)
                {
                    let cargo_count: u64 = self
                        .resource_contents
                        .iter()
                        .find(|(key, _)|**key == id)
                        .map(|(_, num)|*num)
                        .expect("Failed to find contents entry for resource that is supposed to be allowed!");
                    let remainder = value.saturating_sub(cargo_count);
                    *self.resource_contents.get_mut(&id).unwrap() -= value - remainder;
                    Some(GenericCargo::Resource {
                        id: id,
                        value: (value - remainder),
                    })
                } else {
                    None
                }
            }
            GenericCargo::ShipInstance(id) => {
                if self.ship_contents.remove(&id) {
                    Some(cargo)
                } else {
                    None
                }
            }
        }
    }
}

//a given shared stockpile type has its contents shared between all instances of itself; it does not produce any salience propagation
//previously this was an atomic I64 and we don't remember why, but an atomic U64 seems better and hopefully it doesn't subtly break something
#[derive(Debug, Clone)]
pub struct SharedStockpile {
    pub resourcetype: Key<Resource>,
    pub contents: Arc<AtomicU64>,
    pub rate: u64,
    pub capacity: u64,
}

impl Stockpileness for SharedStockpile {
    fn get_resource_contents(&self) -> HashMap<Key<Resource>, u64> {
        iter::once((
            self.resourcetype,
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        HashSet::new()
    }
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        iter::once((
            CollatedCargo::Resource(self.resourcetype),
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        if cargo == self.resourcetype {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        0
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.contents.load(atomic::Ordering::SeqCst)
            * root.resources.get(self.resourcetype).unwrap().cargovol
    }
    fn get_allowed(&self) -> Option<(Vec<Key<Resource>>, Vec<Key<ShipClass>>)> {
        Some((vec![self.resourcetype], Vec::new()))
    }
    fn get_resource_supply(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        if resourceid == self.resourcetype {
            self.contents.load(atomic::Ordering::SeqCst)
                * root.resources.get(resourceid).unwrap().cargovol
        } else {
            0
        }
    }
    fn get_resource_demand(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        0
    }
    fn get_shipclass_supply(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        0
    }
    fn get_shipclass_demand(&self, root: &Root, shipclassid: Key<ShipClass>) -> u64 {
        0
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo {
            GenericCargo::Resource { id, value } => {
                let cargo_vol = root.resources.get(id).unwrap().cargovol;
                if id == self.resourcetype {
                    let count_capacity = self.capacity / cargo_vol;
                    let remainder = value.saturating_sub(
                        count_capacity - self.contents.load(atomic::Ordering::SeqCst),
                    );
                    self.contents
                        .fetch_add(value - remainder, atomic::Ordering::SeqCst);
                    Some(GenericCargo::Resource {
                        id: id,
                        value: remainder,
                    })
                } else {
                    //this will just hand the cargo back to whoever was trying to put it in
                    Some(cargo)
                }
            }
            _ => Some(cargo),
        }
    }
    fn remove(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo {
            GenericCargo::Resource { id, value } => {
                if id == self.resourcetype {
                    let remainder =
                        value.saturating_sub(self.contents.load(atomic::Ordering::SeqCst));
                    self.contents
                        .fetch_sub(value - remainder, atomic::Ordering::SeqCst);
                    Some(GenericCargo::Resource {
                        id: id,
                        value: (value - remainder),
                    })
                } else {
                    //we can't get out something the stockpile can't hold
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HangarClass {
    pub id: Key<HangarClass>,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub capacity: u64,                       //total volume the hangar can hold
    pub target: u64, //volume the hangar wants to hold; this is usually either equal to capacity (for carriers) or zero (for shipyard outputs)
    pub allowed: Vec<Key<ShipClass>>, //which shipclasses this hangar can hold
    pub ideal: HashMap<Key<ShipClass>, u64>, //how many of each ship type the hangar wants
    pub launch_volume: u64, //how much volume the hangar can launch at one time in battle
    pub launch_interval: u64, //time between launches in battle
    pub propagate: bool, //whether or not hangar generates saliences
}

impl HangarClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.ideal
            .iter()
            .map(|(k, v)| root.shipclasses.get(*k).unwrap().get_ideal_strength(root) * v)
            .sum()
    }
    pub fn instantiate(&self) -> HangarInstance {
        HangarInstance {
            hangarclass: self.id,
            visibility: self.visibility,
            capacity: self.capacity,
            target: self.target,
            allowed: self.allowed.clone(),
            ideal: self.ideal.clone(),
            contents: HashSet::new(),
            launch_volume: self.launch_volume,
            launch_interval: self.launch_interval,
            propagate: self.propagate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HangarInstance {
    pub hangarclass: Key<HangarClass>,
    pub visibility: bool,
    pub capacity: u64,                       //total volume the hangar can hold
    pub target: u64, //volume the hangar wants to hold; this is usually either equal to capacity (for carriers) or zero (for shipyard outputs)
    pub allowed: Vec<Key<ShipClass>>, //which shipclasses this hangar can hold
    pub ideal: HashMap<Key<ShipClass>, u64>, //how many of each ship type the hangar wants
    pub contents: HashSet<Key<ShipInstance>>,
    pub launch_volume: u64, //how much volume the hangar can launch at one time in battle
    pub launch_interval: u64, //time between launches in battle
    pub propagate: bool,
}

impl HangarInstance {
    pub fn get_strength(&self, root: &Root, time: u64) -> u64 {
        let contents_strength = self
            .contents
            .iter()
            .map(|s| root.shipinstances.get(*s).unwrap().get_strength(root, time))
            .sum::<u64>() as f32;
        let contents_vol = self
            .contents
            .iter()
            .map(|s| {
                root.shipclasses
                    .get(root.shipinstances.get(*s).unwrap().shipclass)
                    .unwrap()
                    .cargovol
                    .unwrap()
            })
            .sum::<u64>() as f32;
        //we calculate how much of its complement the hangar can launch during a battle a certain number of seconds long
        let launch_mod = (contents_vol / self.launch_volume as f32)
            * (time as f32 / self.launch_interval as f32);
        (contents_strength * launch_mod) as u64
    }
    pub fn get_shipclass_num(&self, root: &Root, shipclass: Key<ShipClass>) -> u64 {
        self.contents
            .iter()
            .filter(|s| root.shipinstances.get(**s).unwrap().shipclass == shipclass)
            .collect::<Vec<_>>()
            .len()
            .try_into()
            .unwrap()
    }
    pub fn get_shipclass_supply(&self, root: &Root, shipclass: Key<ShipClass>) -> u64 {
        self.contents
            .iter()
            .filter(|s| root.shipinstances.get(**s).unwrap().shipclass == shipclass)
            .map(|_| root.shipclasses.get(shipclass).unwrap().cargovol.unwrap())
            .sum()
    }
    pub fn get_shipclass_demand(&self, root: &Root, shipclass: Key<ShipClass>) -> u64 {
        let ideal_num = self.ideal.get(&shipclass).unwrap_or(&0);
        ideal_num.saturating_sub(self.get_shipclass_num(root, shipclass))
            * root.shipclasses.get(shipclass).unwrap().cargovol.unwrap()
    }
}

fn collapse_cargo_maps(vec: &Vec<HashMap<CollatedCargo, u64>>) -> HashMap<CollatedCargo, u64> {
    vec.iter()
        .flatten()
        .fold(HashMap::new(), |mut acc, (cargo, num)| {
            *acc.entry(*cargo).or_insert(0) += num;
            acc
        })
}

pub trait ResourceProcess {
    fn get_state(&self) -> FactoryState;
    fn get_resource_supply_total(&self, root: &Root, resourceid: Key<Resource>) -> u64;
    fn get_resource_demand_total(&self, root: &Root, resourceid: Key<Resource>) -> u64;
    fn get_target_total(&self) -> u64;
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64>;
    fn get_output_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64;
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineClass {
    pub id: Key<EngineClass>,
    pub visiblename: String,
    pub description: String,
    pub basehealth: Option<u64>,
    pub toughnessscalar: f32,
    pub visibility: bool,
    pub inputs: Vec<UnipotentResourceStockpile>,
    pub forbidden_nodeflavors: Vec<Key<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Key<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl EngineClass {
    pub fn instantiate(&self) -> EngineInstance {
        EngineInstance {
            engineclass: self.id,
            basehealth: self.basehealth,
            health: self.basehealth,
            toughnessscalar: self.toughnessscalar,
            visibility: self.visibility,
            inputs: self.inputs.clone(),
            forbidden_nodeflavors: self.forbidden_nodeflavors.clone(),
            forbidden_edgeflavors: self.forbidden_edgeflavors.clone(),
            speed: self.speed,
            cooldown: self.cooldown,
            last_move_turn: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineInstance {
    engineclass: Key<EngineClass>,
    basehealth: Option<u64>,
    health: Option<u64>,
    toughnessscalar: f32,
    visibility: bool,
    inputs: Vec<UnipotentResourceStockpile>,
    forbidden_nodeflavors: Vec<Key<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    forbidden_edgeflavors: Vec<Key<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    cooldown: u64,
    last_move_turn: u64,
}

impl ResourceProcess for EngineInstance {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutputInstance sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }
    fn get_resource_supply_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resourceid))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        HashMap::new()
    }
    fn get_output_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        0
    }
}

impl EngineInstance {
    fn check_engine(
        &self,
        root: &Root,
        location: Key<Node>,
        destinations: &Vec<Key<Node>>,
    ) -> Option<(Vec<Key<Node>>, u64)> {
        if (self.health != Some(0))
            && (root.turn - self.last_move_turn > self.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            let viable = destinations
                .iter()
                .filter(|id| {
                    let node = root.nodes.get(**id).unwrap();
                    !self.forbidden_nodeflavors.contains(&node.flavor)
                        && !self
                            .forbidden_edgeflavors
                            .contains(root.edges.get(&(location, **id)).unwrap())
                })
                .map(|id| *id)
                .collect();
            Some((viable, self.speed))
        } else {
            None
        }
    }
    //this is run once per turn for a given engine; it checks to see if the engine has enough resources to run this turn, whether it's already been run, and whether it's off cooldown
    //then consumes stockpile resources, and sets the engine's moves left for the turn to equal its speed
    //we'll need to reset movement_left to max at the start of the turn
    fn process_engine(
        &mut self,
        root: &Root,
        location: Key<Node>,
        destinations: &Vec<Key<Node>>,
    ) -> Option<(Vec<Key<Node>>, u64)> {
        if (self.health != Some(0))
            && (root.turn - self.last_move_turn > self.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.last_move_turn = root.turn;
            let viable = destinations
                .iter()
                .filter(|id| {
                    let node = root.nodes.get(**id).unwrap();
                    !self.forbidden_nodeflavors.contains(&node.flavor)
                        && !self
                            .forbidden_edgeflavors
                            .contains(root.edges.get(&(location, **id)).unwrap())
                })
                .map(|id| *id)
                .collect();
            Some((viable, self.speed))
        } else {
            None
        }
    }
    fn nav_check(
        forbidden_nodeflavors: Vec<Key<NodeFlavor>>,
        forbidden_edgeflavors: Vec<Key<EdgeFlavor>>,
        location: Key<Node>,
        destination: Key<Node>,
        root: &Root,
    ) -> bool {
        !forbidden_nodeflavors.contains(&root.nodes.get(destination).unwrap().flavor)
            && !forbidden_edgeflavors.contains(root.edges.get(&(location, destination)).unwrap())
        //NOTE: I'd like to have this unwrap make the check fail instead of crashing.
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepairerClass {
    pub id: Key<RepairerClass>,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentResourceStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
}

impl RepairerClass {
    pub fn instantiate(&self) -> RepairerInstance {
        RepairerInstance {
            repairerclass: self.id,
            visibility: self.visibility,
            inputs: self.inputs.clone(),
            repair_points: self.repair_points,
            repair_factor: self.repair_factor,
            engine_repair_points: self.engine_repair_points,
            engine_repair_factor: self.engine_repair_factor,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepairerInstance {
    repairerclass: Key<RepairerClass>,
    visibility: bool,
    inputs: Vec<UnipotentResourceStockpile>,
    repair_points: i64,
    repair_factor: f32,
    engine_repair_points: i64,
    engine_repair_factor: f32,
}

impl ResourceProcess for RepairerInstance {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutputInstance sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }
    fn get_resource_supply_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resourceid))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        HashMap::new()
    }
    fn get_output_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactoryClass {
    pub id: Key<FactoryClass>,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    pub fn instantiate(&self) -> FactoryInstance {
        FactoryInstance {
            factoryclass: self.id,
            visibility: self.visibility,
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactoryInstance {
    //this is an actual factory, derived from a factory class
    factoryclass: Key<FactoryClass>,
    visibility: bool,
    inputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset consumption
    outputs: Vec<UnipotentResourceStockpile>, //the data for the factory's asset production
}

impl ResourceProcess for FactoryInstance {
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
    fn get_resource_supply_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.outputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_supply(root, resourceid))
            .sum::<u64>()
    }
    fn get_resource_demand_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resourceid))
            .sum::<u64>()
            + self
                .outputs
                .iter()
                .filter(|sp| sp.propagate)
                .map(|sp| sp.get_resource_demand(root, resourceid))
                .sum::<u64>()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum::<u64>()
            + self.outputs.iter().map(|sp| sp.target).sum::<u64>()
    }
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        collapse_cargo_maps(
            &self
                .outputs
                .iter()
                .map(|sp| sp.collate_contents(root))
                .collect(),
        )
    }
    fn get_output_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        self.outputs
            .iter()
            .map(|sp| sp.get_resource_num(root, cargo))
            .sum()
    }
}

impl FactoryInstance {
    //we take an active factory and update all its inputs and outputs to add or remove resources
    fn process(&mut self, location_efficiency: f64) {
        if let FactoryState::Active = self.get_state() {
            //dbg!("Factory is active.");
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

#[derive(Debug, Clone, PartialEq)]
pub struct ShipyardClass {
    pub id: Key<ShipyardClass>,
    pub visiblename: Option<String>,
    pub description: Option<String>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentResourceStockpile>,
    pub outputs: HashMap<Key<ShipClass>, u64>,
    pub constructrate: u64,
    pub efficiency: f64,
}

impl ShipyardClass {
    pub fn instantiate(&self) -> ShipyardInstance {
        ShipyardInstance {
            shipyardclass: self.id,
            visibility: self.visibility,
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
    shipyardclass: Key<ShipyardClass>,
    visibility: bool,
    inputs: Vec<UnipotentResourceStockpile>,
    outputs: HashMap<Key<ShipClass>, u64>,
    constructpoints: u64,
    constructrate: u64,
    efficiency: f64,
}

impl ResourceProcess for ShipyardInstance {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutputInstance sufficiency method defined earlier
        if input_is_good {
            FactoryState::Active
        } else {
            FactoryState::Stalled
        }
    }

    fn get_resource_supply_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        0
    }

    fn get_resource_demand_total(&self, root: &Root, resourceid: Key<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resourceid))
            .sum()
    }

    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }

    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        HashMap::new()
    }

    fn get_output_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        0
    }
}

impl ShipyardInstance {
    fn process(&mut self, location_efficiency: f64) {
        if let FactoryState::Active = self.get_state() {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.constructpoints += (self.constructrate as f64 * location_efficiency) as u64;
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
        let cost = shipclasstable.get(*shipclassid).unwrap().basestrength;
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ShipLocationFlavor {
    Node(Key<Node>),
    Fleet(Key<FleetInstance>),
    Host(HostFlavor),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum HostFlavor {
    Garrison(Key<Node>),
    Carrier(Key<ShipInstance>),
}

impl HostFlavor {
    pub fn get_node(&self, root: &Root) -> Key<Node> {
        match self {
            HostFlavor::Garrison(id) => *id,
            HostFlavor::Carrier(id) => root.shipinstances.get(*id).unwrap().get_node(root),
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
                        .get(&shipinstancetable.get(id).unwrap().shipclass)
                        .unwrap()
                        .cargovol
                        .unwrap()
                })
                .sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShipClass {
    pub id: Key<ShipClass>,
    pub visiblename: String,
    pub description: String,
    pub basehull: u64,     //how many hull hitpoints this ship has by default
    pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub hangarvol: Option<u64>,
    pub cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
    pub stockpiles: Vec<PluripotentStockpile>,
    pub defaultweapons: Option<HashMap<Key<Resource>, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<Key<HangarClass>>,
    pub engines: Vec<Key<EngineClass>>,
    pub repairers: Vec<Key<RepairerClass>>,
    pub factoryclasslist: Vec<Key<FactoryClass>>,
    pub shipyardclasslist: Vec<Key<ShipyardClass>>,
    pub aiclass: Key<ShipAI>,
    pub defectchance: HashMap<Key<Faction>, f64>,
    pub toughnessscalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub escapescalar: f32, //is added to toughnessscalar in battles where this ship is on the losing side, trying to escape
}

impl ShipClass {
    fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.basestrength
            + self
                .hangars
                .iter()
                .map(|h| root.hangarclasses.get(*h).unwrap().get_ideal_strength(root))
                .sum::<u64>()
    }
    //method to create a ship instance with this ship class
    fn instantiate(
        &self,
        location: ShipLocationFlavor,
        faction: Key<Faction>,
        root: &Root,
    ) -> ShipInstance {
        ShipInstance {
            visiblename: uuid::Uuid::new_v4().to_string(),
            shipclass: self.id,
            basehull: self.basehull,
            hull: self.basehull,
            basestrength: self.basestrength,
            visibility: self.visibility,
            hangarvol: self.hangarvol,
            cargovol: self.cargovol,
            stockpiles: self.stockpiles.clone(),
            efficiency: 1.0,
            hangars: self
                .hangars
                .iter()
                .map(|h| root.hangarclasses.get(*h).unwrap().instantiate())
                .collect(),
            engines: self
                .engines
                .iter()
                .map(|classid| root.engineclasses.get(*classid).unwrap().instantiate())
                .collect(),
            movement_left: 1000000000000,
            repairers: self
                .repairers
                .iter()
                .map(|classid| root.repairerclasses.get(*classid).unwrap().instantiate())
                .collect(),
            factoryinstancelist: self
                .factoryclasslist
                .iter()
                .map(|classid| root.factoryclasses.get(*classid).unwrap().instantiate())
                .collect(),
            shipyardinstancelist: self
                .shipyardclasslist
                .iter()
                .map(|classid| root.shipyardclasses.get(*classid).unwrap().instantiate())
                .collect(),
            location,
            allegiance: faction,
            experience: 1.0,
            objectives: Vec::new(),
            aiclass: self.aiclass,
            defectchance: self.defectchance.clone(),
            toughnessscalar: self.toughnessscalar,
            escapescalar: self.escapescalar,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShipInstance {
    pub visiblename: String,
    pub shipclass: Key<ShipClass>, //which class of ship this is
    pub basehull: u64,             //how many hull hitpoints this ship has by default
    pub hull: u64,                 //how many hitpoints the ship has
    pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub hangarvol: Option<u64>,
    pub cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
    pub stockpiles: Vec<PluripotentStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<HangarInstance>,
    pub engines: Vec<EngineInstance>,
    pub movement_left: u64, //starts at one trillion each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<RepairerInstance>,
    pub factoryinstancelist: Vec<FactoryInstance>,
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub location: ShipLocationFlavor, //where the ship is -- a node if it's unaffiliated, a fleet if it's in one
    pub allegiance: Key<Faction>,     //which faction this ship belongs to
    pub experience: f32, //XP gained by this ship, which affects strength score and in-mission AI class
    pub objectives: Vec<ObjectiveFlavor>,
    pub aiclass: Key<ShipAI>,
    pub defectchance: HashMap<Key<Faction>, f64>,
    pub toughnessscalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub escapescalar: f32, //is added to toughnessscalar in battles where this ship is on the losing side, trying to escape
}

impl ShipInstance {
    pub fn get_daughters(&self, shipinstances: &Table<ShipInstance>) -> Vec<Key<ShipInstance>> {
        self.hangars
            .iter()
            .map(|h| {
                h.contents
                    .iter()
                    .map(|s| {
                        let mut vec = shipinstances.get(*s).unwrap().get_daughters(shipinstances);
                        vec.insert(0, *s);
                        vec
                    })
                    .collect::<Vec<Vec<Key<ShipInstance>>>>()
            })
            .flatten()
            .flatten()
            .collect()
    }
    pub fn get_checked_daughters(
        &self,
        root: &Root,
        destination: Key<Node>,
    ) -> Vec<Key<ShipInstance>> {
        self.hangars
            .iter()
            .map(|h| {
                let (active, passive): (Vec<Key<ShipInstance>>, Vec<Key<ShipInstance>>) =
                    h.contents.iter().partition(|s| {
                        root.shipinstances
                            .get(**s)
                            .unwrap()
                            .nav_check(root, destination)
                    });
                let vec1 = active
                    .iter()
                    .map(|s| {
                        let mut vec1 = root
                            .shipinstances
                            .get(*s)
                            .unwrap()
                            .get_daughters(&root.shipinstances);
                        vec1.insert(0, *s);
                        vec1
                    })
                    .collect::<Vec<Vec<Key<ShipInstance>>>>();
                let vec2 = passive
                    .iter()
                    .map(|s| {
                        let mut vec2 = root
                            .shipinstances
                            .get(*s)
                            .unwrap()
                            .get_checked_daughters(root, destination);
                        vec2.insert(0, *s);
                        vec2
                    })
                    .collect::<Vec<Vec<Key<ShipInstance>>>>();
                vec![vec1, vec2]
            })
            .flatten()
            .flatten()
            .flatten()
            .collect()
    }
    pub fn kill(shipid: Key<ShipInstance>, shipinstances: &mut Table<ShipInstance>) {
        let mut vec = shipinstances
            .get(shipid)
            .unwrap()
            .get_daughters(shipinstances);
        vec.insert(0, shipid);
        vec.iter().for_each(|s| {
            shipinstances.get_mut(*s).unwrap().hull = 0;
        });
    }
    pub fn get_strength(&self, root: &Root, time: u64) -> u64 {
        let daughter_strength = self
            .hangars
            .iter()
            .map(|h| h.get_strength(root, time))
            .sum::<u64>();
        let objective_strength: f32 = self
            .objectives
            .iter()
            .map(|of| of.get_scalars().strengthscalar)
            .product();
        (self.basestrength as f32
            * (self.hull as f32 / self.basehull as f32)
            * self.experience
            * objective_strength) as u64
            + daughter_strength
    }
    pub fn check_engines(
        &self,
        root: &Root,
        location: Key<Node>,
        destinations: &Vec<Key<Node>>,
    ) -> Vec<Key<Node>> {
        if self.movement_left > 0 {
            if let Some((viable, speed)) = self
                .engines
                .iter()
                .find_map(|e| e.check_engine(root, location, destinations))
            {
                viable
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }
    pub fn try_engines(
        &mut self,
        root: &Root,
        location: Key<Node>,
        destinations: &Vec<Key<Node>>,
    ) -> Vec<Key<Node>> {
        if self.movement_left > 0 {
            if let Some((viable, speed)) = self
                .engines
                .iter_mut()
                .find_map(|e| e.process_engine(root, location, destinations))
            {
                self.movement_left.saturating_sub(1000000000000 / speed);
                viable
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }
    pub fn reset_movement(&mut self) {
        self.movement_left = 1000000000000;
    }
    pub fn repair(&mut self, engineclasses: &Table<EngineClass>) {
        self.repairers
            .iter()
            .filter(|rp| rp.get_state() == FactoryState::Active)
            .for_each(|rp| {
                self.hull = (self.hull as i64
                    + rp.repair_points
                    + (self.basehull as f32 * rp.repair_factor) as i64)
                    .clamp(0, self.basehull as i64) as u64;
                self.engines
                    .iter_mut()
                    .filter(|e| e.health.is_some())
                    .for_each(|e| {
                        (e.health.unwrap() as i64
                            + rp.engine_repair_points
                            + (e.basehealth.unwrap() as f32 * rp.engine_repair_factor) as i64)
                            .clamp(0, e.basehealth.unwrap() as i64) as u64;
                    })
            })
    }
    pub fn get_resource_num(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_resource_num(root, cargo))
            .sum::<u64>()
            + self
                .factoryinstancelist
                .iter()
                .map(|f| f.get_output_resource_num(root, cargo))
                .sum::<u64>()
    }
    pub fn get_resource_supply(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        self.stockpiles
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_supply(root, cargo))
            .sum::<u64>()
            + self
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_supply_total(root, cargo))
                .sum::<u64>()
    }
    pub fn get_resource_demand(&self, root: &Root, cargo: Key<Resource>) -> u64 {
        self.stockpiles
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, cargo))
            .sum::<u64>()
            + self
                .engines
                .iter()
                .map(|e| e.get_resource_demand_total(root, cargo))
                .sum::<u64>()
            + self
                .repairers
                .iter()
                .map(|r| r.get_resource_demand_total(root, cargo))
                .sum::<u64>()
            + self
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_demand_total(root, cargo))
                .sum::<u64>()
            + self
                .shipyardinstancelist
                .iter()
                .map(|s| s.get_resource_demand_total(root, cargo))
                .sum::<u64>()
    }
    pub fn get_shipclass_num(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_num(root, cargo))
            .sum::<u64>()
            + self
                .hangars
                .iter()
                .map(|h| h.get_shipclass_num(root, cargo))
                .sum::<u64>()
    }
    pub fn get_shipclass_supply(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_supply(root, cargo))
            .sum::<u64>()
            + self
                .hangars
                .iter()
                .map(|h| h.get_shipclass_supply(root, cargo))
                .sum::<u64>()
            + if self.shipclass == cargo {
                root.shipclasses
                    .get(self.shipclass)
                    .unwrap()
                    .cargovol
                    .unwrap()
            } else {
                0
            }
    }
    pub fn get_shipclass_demand(&self, root: &Root, cargo: Key<ShipClass>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_demand(root, cargo))
            .sum::<u64>()
            + self
                .hangars
                .iter()
                .map(|h| h.get_shipclass_demand(root, cargo))
                .sum::<u64>()
    }
    pub fn get_resource_demand_ratio(&self, root: &Root, resourceid: Key<Resource>) -> f32 {
        let demand_total = self
            .stockpiles
            .iter()
            .map(|sp| sp.get_resource_demand(root, resourceid))
            .sum::<u64>() as f32;
        let target_total = self.stockpiles.iter().map(|sp| sp.target).sum::<u64>() as f32;
        assert!(demand_total < target_total);
        demand_total / target_total
    }
    pub fn get_shipclass_demand_ratio(&self, root: &Root, shipclassid: Key<ShipClass>) -> f32 {
        let demand_total = self
            .stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_demand(root, shipclassid))
            .sum::<u64>() as f32;
        let target_total = self.stockpiles.iter().map(|sp| sp.target).sum::<u64>() as f32;
        assert!(demand_total < target_total);
        demand_total / target_total
    }
    pub fn is_in_node(&self, root: &Root) -> bool {
        match self.location {
            ShipLocationFlavor::Node(_) => true,
            _ => false,
        }
    }
    pub fn is_in_fleet(&self, root: &Root) -> bool {
        match self.location {
            ShipLocationFlavor::Node(_) => false,
            ShipLocationFlavor::Fleet(_) => true,
            ShipLocationFlavor::Host(hf) => match hf {
                HostFlavor::Garrison(_) => false,
                HostFlavor::Carrier(k) => root.shipinstances.get(k).unwrap().is_in_fleet(root),
            },
        }
    }
    pub fn get_fleet(&self, root: &Root) -> Option<Key<FleetInstance>> {
        match self.location {
            ShipLocationFlavor::Node(_) => None,
            ShipLocationFlavor::Fleet(k) => Some(k),
            ShipLocationFlavor::Host(hf) => match hf {
                HostFlavor::Garrison(_) => None,
                HostFlavor::Carrier(k) => root.shipinstances.get(k).unwrap().get_fleet(root),
            },
        }
    }
    pub fn get_carrier(selfid: Key<ShipInstance>, root: &Root) -> Key<ShipInstance> {
        match root.shipinstances.get(selfid).unwrap().location {
            ShipLocationFlavor::Node(_) => selfid,
            ShipLocationFlavor::Fleet(_) => selfid,
            ShipLocationFlavor::Host(hf) => match hf {
                HostFlavor::Garrison(_) => selfid,
                HostFlavor::Carrier(k) => ShipInstance::get_carrier(k, root),
            },
        }
    }
    //determines which node the ship is in
    //a ship can be in a number of places which aren't directly in a node, but all of them cash out to a node eventually
    pub fn get_node(&self, root: &Root) -> Key<Node> {
        match self.location {
            ShipLocationFlavor::Node(id) => id,
            ShipLocationFlavor::Fleet(id) => root.fleetinstances.get(id).unwrap().location,
            ShipLocationFlavor::Host(flavor) => flavor.get_node(root),
        }
    }
    pub fn nav_check(&self, root: &Root, destination: Key<Node>) -> bool {
        !self
            .check_engines(root, self.get_node(root), &vec![destination])
            .is_empty()
    }
    pub fn navigate(
        //used for ships which are operating independently
        //this method determines which of the current node's neighbors is most desirable
        &self,
        root: &Root,
        destinations: &Vec<Key<Node>>,
    ) -> Option<Key<Node>> {
        let location: Key<Node> = self.get_node(root);
        //we iterate over the destinations to determine which neighbor is most desirable
        destinations
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .max_by_key(|nodeid| {
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                let resource_demand_value: f32 = root
                    .shipais
                    .get(self.aiclass)
                    .unwrap()
                    .resource_attract
                    .iter()
                    .map(|(resourceid, scalar)| {
                        let demand = root.globalsalience.resourcesalience[self.allegiance.index]
                            [resourceid.index][nodeid.index][0];
                        let supply = root.globalsalience.resourcesalience[self.allegiance.index]
                            [resourceid.index][nodeid.index][1];
                        //let cargo = self.stockpiles.iter().map(|x|)
                        (demand - supply)
                            * (self.get_resource_num(root, *resourceid) as f32
                                * (root.resources.get(*resourceid).unwrap().cargovol) as f32)
                            * scalar
                    })
                    .sum();
                let resource_supply_value: f32 = root
                    .shipais
                    .get(self.aiclass)
                    .unwrap()
                    .resource_attract
                    .iter()
                    .map(|(resourceid, scalar)| {
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not position.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        let demand = root.globalsalience.resourcesalience[self.allegiance.index]
                            [resourceid.index][location.index][0];
                        let supply = root.globalsalience.resourcesalience[self.allegiance.index]
                            [resourceid.index][nodeid.index][1];
                        supply * demand * self.get_resource_demand_ratio(root, *resourceid) * scalar
                    })
                    .sum();
                let shipclass_demand_value: f32 = root
                    .shipais
                    .get(self.aiclass)
                    .unwrap()
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        let demand = root.globalsalience.shipclasssalience[self.allegiance.index]
                            [shipclassid.index][nodeid.index][0];
                        let supply = root.globalsalience.shipclasssalience[self.allegiance.index]
                            [shipclassid.index][nodeid.index][1];
                        (demand - supply)
                            * (self.get_shipclass_num(root, *shipclassid) as f32
                                * root
                                    .shipclasses
                                    .get(*shipclassid)
                                    .unwrap()
                                    .cargovol
                                    .unwrap_or(0) as f32)
                            * scalar
                    })
                    .sum();
                //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
                let shipclass_supply_value: f32 = root
                    .shipais
                    .get(self.aiclass)
                    .unwrap()
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not location.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        let demand = root.globalsalience.shipclasssalience[self.allegiance.index]
                            [shipclassid.index][location.index][0];
                        let supply = root.globalsalience.shipclasssalience[self.allegiance.index]
                            [shipclassid.index][nodeid.index][1];
                        supply
                            * demand
                            * self.get_shipclass_demand_ratio(root, *shipclassid)
                            * scalar
                    })
                    .sum();
                //this checks how much demand there is in the node for ships of the subject ship's class
                let ship_value_specific: f32 = root.globalsalience.shipclasssalience
                    [self.allegiance.index][self.shipclass.index][nodeid.index][0]
                    * root
                        .shipais
                        .get(self.aiclass)
                        .unwrap()
                        .ship_attract_specific;
                //oh, THIS is why we needed the placeholder ship class
                //this checks how much demand there is in the node for ships in general
                let ship_value_generic: f32 = root.globalsalience.shipclasssalience
                    [self.allegiance.index][0][nodeid.index][0]
                    * root.shipais.get(self.aiclass).unwrap().ship_attract_generic;

                NotNan::new(
                    resource_demand_value
                        + resource_supply_value
                        + shipclass_demand_value
                        + shipclass_supply_value
                        + ship_value_specific
                        + ship_value_generic,
                )
                .unwrap()
            })
            .copied()
        //if this doesn't work for some reason, we return None
    }
    //this moves a ship across one edge so long as it has a functioning engine, draining fuel from the engines it uses
    //in turn processing, we'll need to repeat traversal as long as it continues returning true
    pub fn traverse(&mut self, location: Key<Node>, root: &Root) -> Option<Key<Node>> {
        let neighbors = root.neighbors.get(&location).unwrap();
        let destinations = self.try_engines(root, location, &neighbors);
        self.navigate(root, &destinations)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FleetClass {
    pub id: Key<FleetClass>,
    pub visiblename: String,
    pub description: String,
    pub strengthmod: (f32, u64),
    pub visibility: bool,
    pub fleetconfig: HashMap<Key<ShipClass>, u64>,
    pub defectchance: HashMap<Key<Faction>, f64>,
    pub disbandthreshold: f32,
}

impl FleetClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.fleetconfig
            .iter()
            .map(|(k, v)| root.shipclasses.get(*k).unwrap().get_ideal_strength(root) * v)
            .sum()
    }
    pub fn instantiate(
        &self,
        location: Key<Node>,
        faction: Key<Faction>,
        root: &Root,
    ) -> FleetInstance {
        FleetInstance {
            fleetclass: self.id,
            visibility: self.visibility,
            idealstrength: self.get_ideal_strength(root),
            location: location,
            allegiance: faction,
            objectives: Vec::new(),
            phantom: true,
            fleetconfig: self.fleetconfig.clone(),
            defectchance: self.defectchance.clone(),
            disbandthreshold: self.disbandthreshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FleetInstance {
    fleetclass: Key<FleetClass>,
    visibility: bool,
    idealstrength: u64,
    location: Key<Node>,
    allegiance: Key<Faction>,
    objectives: Vec<ObjectiveFlavor>,
    phantom: bool,
    fleetconfig: HashMap<Key<ShipClass>, u64>,
    defectchance: HashMap<Key<Faction>, f64>,
    disbandthreshold: f32,
}

impl FleetInstance {
    pub fn get_daughters(fleetid: Key<FleetInstance>, root: &Root) -> Vec<Key<ShipInstance>> {
        root.shipinstances
            .iter()
            .filter(|(_, s)| s.get_fleet(root) == Some(fleetid))
            .map(|(id, _)| *id)
            .collect()
    }
    pub fn get_strength(fleetid: Key<FleetInstance>, root: &Root, time: u64) -> u64 {
        let fleet = root.fleetinstances.get(fleetid).unwrap();
        let (factor, additive) = root.fleetclasses.get(fleet.fleetclass).unwrap().strengthmod;
        let sum = FleetInstance::get_daughters(fleetid, root)
            .iter()
            .map(|id| {
                root.shipinstances
                    .get(*id)
                    .unwrap()
                    .get_strength(root, time)
            })
            .sum::<u64>();
        (sum as f32 * factor) as u64 + additive
    }
    pub fn disband(fleetid: Key<FleetInstance>, root: &mut Root) {
        let daughters = FleetInstance::get_daughters(fleetid, root);
        for shipid in daughters {
            root.shipinstances.get_mut(shipid).unwrap().location =
                ShipLocationFlavor::Node(root.fleetinstances.get(fleetid).unwrap().location);
        }
    }
    //NOTE: Dummied out currently; needs to be built!
    pub fn nav_check(&self, location: Key<Node>, destination: Key<Node>) -> bool {
        true
    }
    //NOTE: Dummied out currently; needs to be built!
    pub fn navigate(&self, root: &Root) -> Key<Node> {
        Key::<Node>::new_from_index(0)
    }
    //NOTE: Dummied out currently; needs to be built!
    pub fn traverse(&mut self, root: &Root) {}
}

#[derive(Debug)]
pub struct Objective {
    condition: ObjectiveFlavor,
    cost: u64,
}

#[derive(Debug, Copy, Clone)]
pub struct ObjectiveScalars {
    difficulty: f32,
    cost: u64,
    durationscalar: f32,
    strengthscalar: f32,
    toughnessscalar: f32,
    escapescalar: f32,
}

#[derive(Debug, Copy, Clone)]
pub enum ObjectiveFlavor {
    ReachNode {
        scalars: ObjectiveScalars,
        node: Key<Node>,
    },
    ShipDeath {
        scalars: ObjectiveScalars,
        ship: Key<ShipInstance>,
    },
    ShipSafe {
        scalars: ObjectiveScalars,
        ship: Key<ShipInstance>,
        nturns: u64,
    },
    FleetDeath {
        scalars: ObjectiveScalars,
        fleet: Key<FleetInstance>,
    },
    FleetSafe {
        scalars: ObjectiveScalars,
        fleet: Key<FleetInstance>,
        nturns: u64,
        strengthfraction: f32,
    },
    NodeCapture {
        scalars: ObjectiveScalars,
        node: Key<Node>,
    },
    NodeSafe {
        scalars: ObjectiveScalars,
        node: Key<Node>,
        nturns: u64,
    },
    SystemCapture {
        scalars: ObjectiveScalars,
        system: Key<System>,
    },
    SystemSafe {
        scalars: ObjectiveScalars,
        system: Key<System>,
        nturns: u64,
        nodesfraction: f32,
    },
}

impl ObjectiveFlavor {
    pub fn get_scalars(&self) -> ObjectiveScalars {
        match self {
            ObjectiveFlavor::ReachNode { scalars, .. } => *scalars,
            ObjectiveFlavor::ShipDeath { scalars, .. } => *scalars,
            ObjectiveFlavor::ShipSafe { scalars, .. } => *scalars,
            ObjectiveFlavor::FleetDeath { scalars, .. } => *scalars,
            ObjectiveFlavor::FleetSafe { scalars, .. } => *scalars,
            ObjectiveFlavor::NodeCapture { scalars, .. } => *scalars,
            ObjectiveFlavor::NodeSafe { scalars, .. } => *scalars,
            ObjectiveFlavor::SystemCapture { scalars, .. } => *scalars,
            ObjectiveFlavor::SystemSafe { scalars, .. } => *scalars,
        }
    }
}

#[derive(Debug)]
pub struct Operation {
    visiblename: String,
    fleet: Key<FleetInstance>,
    objectives: Vec<Objective>,
}

#[derive(Debug, Clone)]
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
    pub duration_damage_scalar: f32, //multiplier for damage increase as battle duration rises
}

#[derive(Debug, Clone)]
pub struct EngagementPrep {
    attackers: HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    defenders: HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    attacker_reinforcements:
        HashMap<Key<Faction>, Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>>,
    defender_reinforcements:
        HashMap<Key<Faction>, Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>>,
    location: Key<Node>,
    aggressor: Key<Faction>,
}

impl EngagementPrep {
    pub fn engagement_prep(root: &Root, location: Key<Node>, aggressor: Key<Faction>) -> Self {
        let belligerents = Node::get_node_forces(location, root);

        //at present there can be only one attacker faction
        //we figure out which of the belligerents is the aggressor
        //then get the attacker's fleets and ships
        let attackers: HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)> =
            belligerents
                .iter()
                .filter(|(fid, _)| **fid == aggressor)
                .map(|(fid, (fs, ss))| {
                    (
                        *fid,
                        (
                            fs.clone(),
                            ss.iter().map(|s| *s).collect::<Vec<Key<ShipInstance>>>(),
                        ),
                    )
                })
                .collect();

        //we do the same for defenders, the only difference being that there can be multiple defender factions
        //we check whether each faction whose assets are currently occupying the location node is at war with the attacker
        let defenders: HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)> =
            belligerents
                .iter()
                .filter(|(fid, _)| **fid != aggressor)
                .filter(|(fid, _)| {
                    root.wars
                        .get(&(**fid.min(&&aggressor), aggressor.max(**fid)))
                        .is_some()
                })
                .map(|(fid, (fs, ss))| {
                    (
                        *fid,
                        (
                            fs.clone(),
                            ss.iter().map(|s| *s).collect::<Vec<Key<ShipInstance>>>(),
                        ),
                    )
                })
                .collect();

        //we go through the location's neighbors and find any ships/fleets allied with the attackers' faction
        //then for each node, we get the scaling factor for travel time -- what percentage of the battle's duration the unit will be present for
        //here we don't strip out ships that are in fleets; we do that later in let-attackers
        let attacker_reinforcements: HashMap<
            Key<Faction>,
            Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = attackers
            .iter()
            .map(|(fid, _)| {
                (
                    *fid,
                    root.neighbors
                        .get(&location)
                        .unwrap()
                        .iter()
                        .map(|n| {
                            (
                                Node::get_distance(*n, location, root),
                                Node::get_node_faction_reinforcements(*n, location, *fid, root),
                            )
                        })
                        .collect(),
                )
            })
            .collect();

        let defender_reinforcements: HashMap<
            Key<Faction>,
            Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = defenders
            .iter()
            .map(|(fid, _)| {
                (
                    *fid,
                    root.neighbors
                        .get(&location)
                        .unwrap()
                        .iter()
                        .map(|n| {
                            (
                                Node::get_distance(*n, location, root),
                                Node::get_node_faction_reinforcements(*n, location, *fid, root),
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        EngagementPrep {
            attackers,
            defenders,
            attacker_reinforcements,
            defender_reinforcements,
            location,
            aggressor,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Engagement {
    visiblename: String,
    attackers: HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    defenders: HashMap<Key<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    aggressor: Key<Faction>,
    objectives: HashMap<Key<Faction>, Vec<ObjectiveFlavor>>,
    location: Key<Node>,
    duration: u64,
    victor: Vec<Key<Faction>>,
    ship_status: HashMap<Key<ShipInstance>, (u64, Vec<u64>, ShipLocationFlavor)>,
    fleet_status: HashMap<Key<FleetInstance>, Key<Node>>,
}

impl Engagement {
    pub fn internal_battle(root: &Root, data: EngagementPrep, scalars: BattleScalars) -> Self {
        //we determine how long the battle lasts
        //taking into account both absolute and relative armada sizes
        //scaled logarithmically according to the specified exponent
        //as well as the scaling factors applied by the objectives of parties involved
        //then we multiply by a random number from a normal distribution
        let attacker_shipcount: i64 = data
            .attackers
            .iter()
            .map(|(_, (_, ss))| ss)
            .flatten()
            .collect::<Vec<_>>()
            .len() as i64;
        let defender_shipcount: i64 = data
            .defenders
            .iter()
            .map(|(_, (_, ss))| ss)
            .flatten()
            .collect::<Vec<_>>()
            .len() as i64;
        let battle_size = (attacker_shipcount + defender_shipcount)
            - (attacker_shipcount - defender_shipcount).abs();

        let objective_duration_scalar: f32 = data
            .attackers
            .iter()
            .map(|(_, (fs, ss))| {
                let mut d = fs
                    .iter()
                    .map(|id| root.fleetinstances.get(*id).unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                        .map(|id| root.shipinstances.get(*id).unwrap().objectives.clone())
                        .flatten()
                        .collect::<Vec<ObjectiveFlavor>>(),
                );
                d
            })
            .flatten()
            .map(|of| of.get_scalars().durationscalar)
            .product::<f32>()
            * data
                .defenders
                .iter()
                .map(|(_, (fs, ss))| {
                    let mut d = fs
                        .iter()
                        .map(|id| root.fleetinstances.get(*id).unwrap().objectives.clone())
                        .flatten()
                        .collect::<Vec<ObjectiveFlavor>>();
                    d.append(
                        &mut ss
                            .iter()
                            .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                            .map(|id| root.shipinstances.get(*id).unwrap().objectives.clone())
                            .flatten()
                            .collect::<Vec<ObjectiveFlavor>>(),
                    );
                    d
                })
                .flatten()
                .map(|of| of.get_scalars().durationscalar)
                .product::<f32>();

        let duration: u64 = (((battle_size as f32).log(root.battlescalars.duration_log_exp)
            + 300.0)
            * objective_duration_scalar
            * Normal::new(1.0, root.battlescalars.duration_dev)
                .unwrap()
                .sample(&mut rand::thread_rng()))
        .clamp(0.0, 2.0) as u64;

        //we take the reinforcement data from our engagementprep and convert the distances to the scaling factor for travel time
        //what percentage of the battle's duration the unit will be present for
        let scaled_attacker_reinforcements: HashMap<
            Key<Faction>,
            Vec<(f32, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = data
            .attacker_reinforcements
            .iter()
            .map(|(fid, vec)| {
                (
                    *fid,
                    vec.iter()
                        .map(|(dist, (fs, ss))| {
                            (
                                (duration.saturating_sub(*dist)) as f32 / duration as f32,
                                //NOTE: Can we get rid of these clones somehow?
                                (fs.clone(), ss.clone()),
                            )
                        })
                        .collect(),
                )
            })
            .collect();

        let scaled_defender_reinforcements: HashMap<
            Key<Faction>,
            Vec<(f32, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = data
            .defender_reinforcements
            .iter()
            .map(|(fid, vec)| {
                (
                    *fid,
                    vec.iter()
                        .map(|(dist, (fs, ss))| {
                            (
                                (duration.saturating_sub(*dist)) as f32 / duration as f32,
                                (fs.clone(), ss.clone()),
                            )
                        })
                        .collect(),
                )
            })
            .collect();

        //we get the strength of all the ships that aren't in fleets
        //followed by the strength of the fleets
        //repeat for reinforcements, then sum everything
        let attacker_strength: u64 = data
            .attackers
            .iter()
            .map(|(_, (fs, ss))| {
                ss.iter()
                    .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                    .map(|shipid| {
                        root.shipinstances
                            .get(*shipid)
                            .unwrap()
                            .get_strength(root, duration)
                    })
                    .sum::<u64>()
                    + fs.iter()
                        .map(|f| FleetInstance::get_strength(*f, root, duration))
                        .sum::<u64>()
            })
            .sum::<u64>()
            + scaled_attacker_reinforcements
                .iter()
                .map(|(_, v)| {
                    v.iter()
                        .map(|(scalar, (fs, ss))| {
                            ((ss.iter()
                                .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                                .map(|s| {
                                    root.shipinstances
                                        .get(*s)
                                        .unwrap()
                                        .get_strength(root, duration)
                                })
                                .sum::<u64>()
                                + fs.iter()
                                    .map(|f| FleetInstance::get_strength(*f, root, duration))
                                    .sum::<u64>()) as f32
                                * scalar) as u64
                        })
                        .sum::<u64>()
                })
                .sum::<u64>();

        let defender_strength: u64 = data
            .defenders
            .iter()
            .map(|(_, (fs, ss))| {
                ss.iter()
                    .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                    .map(|shipid| {
                        root.shipinstances
                            .get(*shipid)
                            .unwrap()
                            .get_strength(root, duration)
                    })
                    .sum::<u64>()
                    + fs.iter()
                        .map(|f| FleetInstance::get_strength(*f, root, duration))
                        .sum::<u64>()
            })
            .sum::<u64>()
            + scaled_defender_reinforcements
                .iter()
                .map(|(fid, v)| {
                    v.iter()
                        .map(|(scalar, (fs, ss))| {
                            ((ss.iter()
                                .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                                .map(|s| {
                                    root.shipinstances
                                        .get(*s)
                                        .unwrap()
                                        .get_strength(root, duration)
                                })
                                .sum::<u64>()
                                + fs.iter()
                                    .map(|f| FleetInstance::get_strength(*f, root, duration))
                                    .sum::<u64>()) as f32
                                * scalar) as u64
                        })
                        .sum::<u64>()
                })
                .sum::<u64>();

        /*
        //NOTE: There might be a more efficient way to do this by combining it with the objective_difficulty gathering
        let objectives = belligerents.iter().fold(HashMap::new(), |mut acc, (fid, (fs, ss))| {
            *acc.entry(fid).or_insert_with(|| {

            })
        })
        */

        //we don't take the objectives of reinforcement units into account
        let attacker_objective_difficulty: f32 = data
            .attackers
            .iter()
            .map(|(_, (fs, ss))| {
                let mut d = fs
                    .iter()
                    .map(|id| root.fleetinstances.get(*id).unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                        .map(|id| root.shipinstances.get(*id).unwrap().objectives.clone())
                        .flatten()
                        .collect::<Vec<ObjectiveFlavor>>(),
                );
                d
            })
            .flatten()
            .map(|of| of.get_scalars().difficulty)
            .product();

        let defender_objective_difficulty: f32 = data
            .attackers
            .iter()
            .map(|(_, (fs, ss))| {
                let mut d = fs
                    .iter()
                    .map(|id| root.fleetinstances.get(*id).unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|s| !root.shipinstances.get(**s).unwrap().is_in_fleet(root))
                        .map(|id| root.shipinstances.get(*id).unwrap().objectives.clone())
                        .flatten()
                        .collect::<Vec<ObjectiveFlavor>>(),
                );
                d
            })
            .flatten()
            .map(|of| of.get_scalars().difficulty)
            .product();

        let attacker_chance: f32 = attacker_strength as f32
            * attacker_objective_difficulty
            * data
                .attackers
                .iter()
                .map(|(id, _)| root.factions.get(*id).unwrap().battlescalar)
                .product::<f32>()
            * Normal::<f32>::new(1.0, root.battlescalars.attacker_chance_dev)
                .unwrap()
                .sample(&mut rand::thread_rng())
                .clamp(0.0, 2.0);

        let defender_chance: f32 = defender_strength as f32
            * defender_objective_difficulty
            * data
                .defenders
                .iter()
                .map(|(id, _)| root.factions.get(*id).unwrap().battlescalar)
                .product::<f32>()
            * Normal::<f32>::new(1.0, root.battlescalars.defender_chance_dev)
                .unwrap()
                .sample(&mut rand::thread_rng())
                .clamp(0.0, 2.0);

        let (victor, victor_strength, victis_strength) = if attacker_chance > defender_chance {
            (
                vec![data.aggressor],
                attacker_strength as f32,
                defender_strength as f32,
            )
        } else {
            (
                data.defenders.iter().map(|(fid, _)| *fid).collect(),
                defender_strength as f32,
                attacker_strength as f32,
            )
        };

        let all_fleets: Vec<Key<FleetInstance>> = data
            .attackers
            .iter()
            .map(|(fid, (fs, ss))| fs.clone())
            .chain(data.defenders.iter().map(|(fid, (fs, ss))| fs.clone())) //NOTE: Can we avoid this clone?
            .chain(
                scaled_attacker_reinforcements
                    .iter()
                    .map(|(fid, vec)| vec.iter().map(|(fid, (fs, ss))| fs.clone()))
                    .flatten(),
            )
            .chain(
                scaled_defender_reinforcements
                    .iter()
                    .map(|(fid, vec)| vec.iter().map(|(fid, (fs, ss))| fs.clone()))
                    .flatten(),
            )
            .flatten()
            .collect();

        let all_ships: Vec<Key<ShipInstance>> = data
            .attackers
            .iter()
            .map(|(fid, (fs, ss))| ss.clone())
            .chain(data.defenders.iter().map(|(fid, (fs, ss))| ss.clone())) //NOTE: Can we avoid this clone?
            .chain(
                scaled_attacker_reinforcements
                    .iter()
                    .map(|(fid, vec)| vec.iter().map(|(fid, (fs, ss))| ss.clone()))
                    .flatten(),
            )
            .chain(
                scaled_defender_reinforcements
                    .iter()
                    .map(|(fid, vec)| vec.iter().map(|(fid, (fs, ss))| ss.clone()))
                    .flatten(),
            )
            .flatten()
            .collect();

        let fleet_status: HashMap<Key<FleetInstance>, Key<Node>> = all_fleets
            .iter()
            .map(|fleetid| {
                (
                    *fleetid,
                    root.fleetinstances.get(*fleetid).unwrap().navigate(root),
                )
            })
            .collect();

        let duration_damage_rand = Normal::<f32>::new(1.0, root.battlescalars.damage_dev)
            .unwrap()
            .sample(&mut rand::thread_rng())
            .clamp(0.0, 1.0);

        //NOTE: Maybe have the lethality scaling over battle duration be logarithmic? Maybe modder-specified?
        let ship_status: HashMap<Key<ShipInstance>, (u64, Vec<u64>, ShipLocationFlavor)> = {
            all_ships
                .iter()
                .map(|shipid| {
                    let ship = root.shipinstances.get(*shipid).unwrap();
                    let rand_factor = Normal::<f32>::new(0.25, root.battlescalars.damage_dev)
                        .unwrap()
                        .sample(&mut rand::thread_rng())
                        .clamp(0.0, 10.0);
                    if !victor.contains(&ship.allegiance) {
                        let new_location = if ship.is_in_node(root) {
                            ShipLocationFlavor::Node(
                                ship.navigate(root, root.neighbors.get(&data.location).unwrap())
                                    .unwrap_or(data.location),
                            )
                        } else {
                            ship.location
                        };
                        //to calculate how much damage the ship takes
                        //we first have a multiplicative damage value, which is:
                        //the ship's maximum health
                        //times the ratio between the enemy's strength and the ship's coalition's strength
                        //times the battle's duration, modified by a modder-specified scalar and a battle-wide random value
                        //times a ship-specific random factor
                        //times the modder-specified multiplier for damage taken by losing ships
                        //
                        //then we add to that an additive damage value, which is:
                        //the modder-defined base damage value
                        //times the strength ratio
                        //times the duration modifier
                        //times the random factor
                        //times the losing-ship multiplier
                        //
                        //then we divide all that by the sum of the ship's toughness and escape scalars
                        let damage = (((ship.basehull as f32
                            * (victor_strength / victis_strength)
                            * (duration as f32
                                * root.battlescalars.duration_damage_scalar
                                * duration_damage_rand)
                            * rand_factor
                            * root.battlescalars.vae_victis)
                            + (root.battlescalars.base_damage
                                * (victor_strength / victis_strength)
                                * (duration as f32
                                    * root.battlescalars.duration_damage_scalar
                                    * duration_damage_rand)
                                * rand_factor
                                * root.battlescalars.vae_victis))
                            / (ship.toughnessscalar + ship.escapescalar))
                            as u64;
                        let engine_damage: Vec<u64> = ship
                            .engines
                            .iter()
                            .filter(|e| e.health.is_some())
                            .map(|e| {
                                ((damage as f32
                                    * Normal::<f32>::new(1.0, root.battlescalars.damage_dev)
                                        .unwrap()
                                        .sample(&mut rand::thread_rng())
                                        .clamp(0.0, 2.0)
                                    * root.battlescalars.engine_damage_scalar)
                                    / e.toughnessscalar) as u64
                            })
                            .collect();
                        (*shipid, (damage, engine_damage, new_location))
                    } else {
                        let new_location = if all_fleets.contains(
                            &ship
                                .get_fleet(root)
                                .unwrap_or(Key::new_from_index(usize::MAX)),
                        ) {
                            ship.location
                        } else {
                            ShipLocationFlavor::Node(data.location)
                        };
                        //we do basically the same thing for winning ships
                        //except that the strength ratio is reversed
                        //we use the damage multiplier for winners instead of losers
                        //and we don't take escapescalar into account
                        let damage = (((ship.basehull as f32
                            * (victis_strength / victor_strength)
                            * (duration as f32
                                * root.battlescalars.duration_damage_scalar
                                * duration_damage_rand)
                            * rand_factor
                            * root.battlescalars.vae_victor)
                            + (root.battlescalars.base_damage
                                * (victis_strength / victor_strength)
                                * (duration as f32
                                    * root.battlescalars.duration_damage_scalar
                                    * duration_damage_rand)
                                * rand_factor
                                * root.battlescalars.vae_victor))
                            / ship.toughnessscalar) as u64;
                        let engine_damage: Vec<u64> = ship
                            .engines
                            .iter()
                            .filter(|e| e.health.is_some())
                            .map(|e| {
                                ((damage as f32
                                    * Normal::<f32>::new(1.0, root.battlescalars.damage_dev)
                                        .unwrap()
                                        .sample(&mut rand::thread_rng())
                                        .clamp(0.0, 2.0)
                                    * root.battlescalars.engine_damage_scalar)
                                    / e.toughnessscalar) as u64
                            })
                            .collect();
                        (*shipid, (damage, engine_damage, new_location))
                    }
                })
                .collect()
        };

        Engagement {
            visiblename: format!(
                "Battle of {}",
                root.nodes.get(data.location).unwrap().visiblename
            ),
            attackers: data.attackers,
            defenders: data.defenders,
            aggressor: data.aggressor,
            objectives: HashMap::new(),
            location: data.location,
            duration: duration,
            victor: victor,
            ship_status: ship_status,
            fleet_status: fleet_status,
        }
    }

    pub fn battle_cleanup(&self, root: &mut Root) {
        if self.victor.contains(&self.aggressor) {
            root.nodes.get_mut(self.location).unwrap().allegiance = self.aggressor
        };
        for (fleetid, l) in &self.fleet_status {
            let fleet = root.fleetinstances.get_mut(*fleetid).unwrap();
            fleet.location = *l;
        }
        for (shipid, (d, ve, l)) in &self.ship_status {
            let ship = root.shipinstances.get_mut(*shipid).unwrap();
            ship.location = *l;
            ship.hull.saturating_sub(*d);
            ve.iter()
                .zip(ship.engines.iter_mut().filter(|e| e.health.is_some()))
                .for_each(|(d, e)| {
                    e.health.unwrap().saturating_sub(*d);
                });
        }
        root.remove_dead();
        root.disband_fleets();
        root.engagements.put(self.clone());
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

    #[derive(Copy, Clone)]
    pub struct Supply {}

    impl Polarity for Supply {}

    #[derive(Copy, Clone)]
    pub struct Demand {}

    impl Polarity for Demand {}
}

pub trait Salience<P: Polarity> {
    const DEG_MULT: f32;
    //this retrieves the value of a specific salience in a specific node
    fn get_value(self, node: (Key<Node>, &Node), faction: Key<Faction>, root: &Root)
        -> Option<f32>;
}

//this method retrieves threat value generated by a given faction
impl Salience<polarity::Supply> for Key<Faction> {
    const DEG_MULT: f32 = 0.5;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Key<Faction>,
        root: &Root,
    ) -> Option<f32> {
        let node_strength: u64 = root.get_node_strength(nodeid, self);
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
        faction: Key<Faction>,
        root: &Root,
    ) -> Option<f32> {
        //NOTE: Currently this does not take input stockpiles of any kind into account. We may wish to change this.
        //we add up all the resource quantity in factory output stockpiles in the node
        let factorysupply: u64 = if node.allegiance == faction {
            node.factoryinstancelist
                .iter()
                .map(|factory| factory.get_resource_supply_total(root, self))
                .sum::<u64>()
        } else {
            0
        };
        //then all the valid resource quantity in ships
        let shipsupply: u64 = root
            .shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(root) == nodeid)
            .filter(|(_, ship)| ship.allegiance == faction)
            .map(|(_, ship)| ship.get_resource_supply(root, self))
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
        faction: Key<Faction>,
        root: &Root,
    ) -> Option<f32> {
        //add up resources from factory input stockpiles in node
        let factorydemand: u64 = if node.allegiance == faction {
            node.factoryinstancelist
                .iter()
                .map(|factory| factory.get_resource_demand_total(root, self))
                .sum::<u64>()
        } else {
            0
        };
        //add up resources from shipyard input stockpiles in node
        let shipyarddemand: u64 = if node.allegiance == faction {
            node.shipyardinstancelist
                .iter()
                .map(|shipyard| shipyard.get_resource_demand_total(root, self))
                .sum::<u64>()
        } else {
            0
        };
        //now we have to look at ships in the node, since they might have stockpiles of their own
        let shipdemand: u64 = root
            .shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(root) == nodeid)
            .filter(|(_, ship)| ship.allegiance == faction)
            .map(|(_, ship)| ship.get_resource_demand(root, self))
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

//this method tells us how much supply there is of a given shipclass in a given node
impl Salience<polarity::Supply> for Key<ShipClass> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Key<Faction>,
        root: &Root,
    ) -> Option<f32> {
        let sum = root
            .shipinstances
            .iter()
            .filter(|(_, s)| s.get_node(root) == nodeid)
            .map(|(_, s)| s.get_shipclass_supply(root, self))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

//this method tells us how much demand there is for a given shipclass in a given node
impl Salience<polarity::Demand> for Key<ShipClass> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Key<Faction>,
        root: &Root,
    ) -> Option<f32> {
        let sum = root
            .shipinstances
            .iter()
            .filter(|(_, s)| s.get_node(root) == nodeid)
            .map(|(_, s)| s.get_shipclass_demand(root, self))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

//TODO: implement supply and demand for shipclasses, and make the logic apply more generally to stockpiles attached to ships

#[derive(Debug, Clone)]
pub struct GlobalSalience {
    pub resourcesalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub shipclasssalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub factionsalience: Vec<Vec<Vec<[f32; 2]>>>,
}

#[derive(Debug)]
pub struct Root {
    pub nodeflavors: Table<NodeFlavor>,
    pub nodes: Table<Node>,
    pub systems: Table<System>,
    pub edgeflavors: Table<EdgeFlavor>,
    pub edges: HashMap<(Key<Node>, Key<Node>), Key<EdgeFlavor>>,
    pub neighbors: HashMap<Key<Node>, Vec<Key<Node>>>,
    pub factions: Table<Faction>,
    pub wars: HashSet<(Key<Faction>, Key<Faction>)>,
    pub resources: Table<Resource>,
    pub hangarclasses: Table<HangarClass>,
    pub engineclasses: Table<EngineClass>,
    pub repairerclasses: Table<RepairerClass>,
    pub factoryclasses: Table<FactoryClass>,
    pub shipyardclasses: Table<ShipyardClass>,
    pub shipais: Table<ShipAI>,
    pub shipclasses: Table<ShipClass>,
    pub shipinstances: Table<ShipInstance>,
    pub shipinstancecounter: usize,
    pub fleetclasses: Table<FleetClass>,
    pub fleetinstances: Table<FleetInstance>,
    pub engagements: Table<Engagement>,
    pub battlescalars: BattleScalars,
    pub globalsalience: GlobalSalience,
    pub turn: u64,
}

impl Root {
    /*pub fn balance_stockpiles(
        &mut self,
        nodeid: Key<Node>,
        faction: Key<Faction>,
        salience_map: Vec<f32>,
    ) {
        let mut node_stockpile = PluripotentStockpile {
            resource_contents: HashMap::new(),
            ship_contents: HashSet::new(),
            allowed: None,
            target: 0,
            capacity: u64::MAX,
            propagate: false,
        };
        let factory_contents: Vec<HashMap<CollatedCargo, u64>> = self
            .nodes
            .get(nodeid)
            .factoryinstancelist
            .iter()
            .map(|x| {
                x.outputs
                    .iter()
                    .map(|stockpile| stockpile.collate_contents(&self))
                    .collect::<Vec<HashMap<CollatedCargo, u64>>>()
            })
            .flatten()
            .collect();
        let ships_in_node: Vec<(Key<ShipInstance>, ShipInstance)> = self
            .shipinstances
            .iter()
            .filter(|(id, ship)| ship.get_node(&self)
            .collect();
        let ship_factory_contents: Vec<HashMap<CollatedCargo, u64>> = ships_in_node
            .iter()
            .map(|(id, ship)| {
                *ship
                    .factoryinstancelist
                    .iter()
                    .map(|factory| {
                        factory
                            .outputs
                            .iter()
                            .map(|stockpile| stockpile.collate_contents(&self))
                            .collect::<Vec<HashMap<CollatedCargo, u64>>>()
                    })
                    .collect::<Vec<Vec<HashMap<CollatedCargo, u64>>>>()
            })
            .flatten()
            .flatten()
            .collect();
        let node_supply = factory_contents.extend(ship_factory_contents);
        //let node_demand =
    }*/

    //this is the method for creating a ship
    //duh
    pub fn create_ship(
        &mut self,
        class_id: Key<ShipClass>,
        location: ShipLocationFlavor,
        faction: Key<Faction>,
    ) -> Key<ShipInstance> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        let new_ship = self
            .shipclasses
            .get(class_id)
            .unwrap()
            .instantiate(location, faction, &self);
        self.shipinstancecounter += 1;
        self.shipinstances.put(new_ship)
    }
    pub fn remove_dead(&mut self) {
        let mut dead: Vec<Key<ShipInstance>> = self
            .shipinstances
            .iter()
            .filter(|(_, s)| s.hull == 0)
            .map(|(k, v)| *k)
            .collect();
        dead.iter_mut()
            .for_each(|id| ShipInstance::kill(*id, &mut self.shipinstances));
        self.shipinstances.retain(|_, s| s.hull > 0);
    }
    pub fn disband_fleets(&mut self) {
        let dead = self
            .fleetinstances
            .iter()
            .filter(|(id, fleet)| {
                let class = self.fleetclasses.get(fleet.fleetclass).unwrap();
                ((FleetInstance::get_strength(**id, self, self.battlescalars.avg_duration) as f32)
                    < (fleet.idealstrength as f32 * class.disbandthreshold))
                    && !fleet.phantom
            })
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in dead {
            FleetInstance::disband(id, self);
        }
        let remaining: Vec<Key<FleetInstance>> = self
            .fleetinstances
            .iter()
            .filter(|(id, fleet)| {
                fleet.phantom || !FleetInstance::get_daughters(**id, self).is_empty()
            })
            .map(|(k, _)| *k)
            .collect();
        self.fleetinstances.retain(|id, _| remaining.contains(id));
    }
    //we get the military strength of a node for a given faction by filtering down the global ship list by node and faction allegiance, then summing their strength values
    fn get_node_strength(&self, nodeid: Key<Node>, faction: Key<Faction>) -> u64 {
        self.shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(&self) == nodeid)
            .filter(|(_, ship)| ship.allegiance == faction)
            //NOTE: here we have 'time' hardcoded as ten minutes; we'll probably want this to be modder-defined
            .map(|(_, ship)| ship.get_strength(&self, self.battlescalars.avg_duration))
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
        //we call get_value on the salience, and return the node id and salience value, while filtering down to only the nodes producing the subject salience
        //Length equals nodes producing subject salience
        let node_initial_salience_map: Vec<(Key<Node>, f32)> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| {
                salience
                    .get_value((id, node), subject_faction, &self)
                    .map(|v| (id, v))
            })
            .collect();
        //this map contains the amount of threat that exists from each faction, in each node, from the perspective of the subject faction
        //Length equals all nodes
        //This is a subjective map for subject faction
        let tagged_threats: Vec<HashMap<Key<Faction>, f32>> = self
            .nodes
            .iter()
            .map(|(_, node)| {
                //we iterate over the node's threat listing, and get the threat for each faction as perceived by the subject faction -- that is, multiplied by the subject faction's relations with that faction
                node.threat
                    .iter()
                    .map(|(f, t)| {
                        let value = t * self
                            .factions
                            .get(subject_faction)
                            .unwrap()
                            .relations
                            .get(f)
                            .unwrap();
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
            .map(|(id, _)| {
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
            (0..n_iters).fold(node_salience_state, |mut state, _| {
                //println!("Completed {} iterations of salience propagation.", n_iter);
                self.edges.iter().for_each(|((a, b), _)| {
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
    pub fn calculate_global_resource_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .iter()
            .map(|(factionid, _)| {
                self.resources
                    .iter()
                    .map(|(resourceid, _)| {
                        let supply = self.calculate_values::<Key<Resource>, polarity::Supply>(
                            *resourceid,
                            *factionid,
                            5,
                        );
                        let demand = self.calculate_values::<Key<Resource>, polarity::Demand>(
                            *resourceid,
                            *factionid,
                            5,
                        );
                        supply
                            .iter()
                            .zip(demand.iter())
                            .map(|(s, d)| [*s, *d])
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn calculate_global_shipclass_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .iter()
            .map(|(factionid, _)| {
                self.shipclasses
                    .iter()
                    .map(|(shipclassid, _)| {
                        let supply = self.calculate_values::<Key<ShipClass>, polarity::Supply>(
                            *shipclassid,
                            *factionid,
                            5,
                        );
                        let demand = self.calculate_values::<Key<ShipClass>, polarity::Demand>(
                            *shipclassid,
                            *factionid,
                            5,
                        );
                        supply
                            .iter()
                            .zip(demand.iter())
                            .map(|(s, d)| [*s, *d])
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn process_turn(&mut self) {
        //NOTE: Scaffolding for turn structure

        //reset all ships' engines

        //run all ship repairers

        //process all factories

        //process all shipyards

        //create queued ships

        //propagate saliences, create salience map

        //run operation management logic

        //move ships, one edge at a time
        //running battle checks and stockpile balancing with each traversal

        //move fleets, one edge at a time
        //running battle checks and stockpile balancing with each traversal

        //run diplomacy logic

        //run defection logic

        //transmit root data to frontend

        //we run the factory process for all factories attached to nodes, so that they produce and consume resources
        self.nodes.iter_mut().for_each(|(_, node)| {
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

        for _ in 0..10 {
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
