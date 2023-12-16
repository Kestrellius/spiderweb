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
use std::sync::RwLock;
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

#[derive(Debug, Clone)]
pub struct Table<T> {
    pub next_index: usize,
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

#[derive(Debug, Clone)]
pub struct Config {
    pub saliencescalars: SalienceScalars,
    pub entityscalars: EntityScalars,
    pub battlescalars: BattleScalars,
}

#[derive(Debug, Clone)]
pub struct SalienceScalars {
    pub faction_deg_mult: f32,
    pub resource_deg_mult: f32,
    pub shipclass_deg_mult: f32,
    pub faction_prop_iters: u64, //number of edges across which this salience will propagate during a turn
    pub resource_prop_iters: u64,
    pub shipclass_prop_iters: u64,
}

#[derive(Debug, Clone)]
pub struct EntityScalars {
    pub defect_escape_scalar: f32,
    pub victor_morale_scalar: f32,
    pub victis_morale_scalar: f32,
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

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct NodeFlavor {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub visiblename: String, //location name as shown to player
    pub system: Key<System>, //system in which node is located; this is used to generate all-to-all in-system links
    pub position: [i64; 3], //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    pub description: String,
    pub visibility: bool,
    pub flavor: Arc<NodeFlavor>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub factoryinstancelist: Vec<FactoryInstance>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub environment: String, //name of the FRED environment to use for missions set in this node
    pub bitmap: Option<(String, f32)>, //name of the bitmap used to depict this node in other nodes' FRED environments, and a size factor
    pub allegiance: Arc<Faction>,      //faction that currently holds the node
    pub efficiency: f32, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub threat: HashMap<Arc<Faction>, f32>,
    pub already_balanced: bool,
}

impl Node {
    pub fn get_node_forces(
        node: Key<Node>,
        root: &Root,
    ) -> HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)> {
        root.factions
            .iter()
            .map(|faction| {
                let ships: Vec<Key<ShipInstance>> = root
                    .shipinstances
                    .iter()
                    .filter(|(_, ship)| ship.allegiance == *faction)
                    .filter(|(_, ship)| ship.get_node(root) == node)
                    .map(|(shipid, _)| *shipid)
                    .collect();
                let fleets: Vec<Key<FleetInstance>> = root
                    .fleetinstances
                    .iter()
                    .filter(|(_, fleet)| fleet.allegiance == *faction)
                    .filter(|(_, fleet)| fleet.location == node)
                    .map(|(fleetid, _)| *fleetid)
                    .collect();
                (faction.clone(), (fleets, ships))
            })
            .filter(|(_, (_, ships))| ships.len() > 0)
            .collect()
    }
    pub fn get_node_factions(nodeid: Key<Node>, root: &Root) -> Vec<Arc<Faction>> {
        root.factions
            .iter()
            .filter(|faction| {
                !root
                    .shipinstances
                    .iter()
                    .filter(|(_, ship)| ship.allegiance == **faction)
                    .filter(|(_, ship)| ship.get_node(root) == nodeid)
                    .collect::<Vec<_>>()
                    .is_empty()
            })
            .cloned()
            .collect()
    }
    pub fn get_node_faction_reinforcements(
        location: Key<Node>,
        destination: Key<Node>,
        factionid: Arc<Faction>,
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
                    .nav_check(root, destination)
                    .is_some()
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
                let mut vec = ship.get_daughters(&root.shipinstances, &root.hangarinstances);
                vec.insert(0, **shipid);
                vec
            })
            .flatten()
            .collect();
        let ships_in_node: Vec<Key<ShipInstance>> = relevant_ships
            .iter()
            .filter(|(_, ship)| ship.is_in_node(root))
            .filter(|(_, ship)| ship.nav_check(root, vec![destination]))
            .map(|(shipid, ship)| {
                let mut vec = ship.get_daughters(&root.shipinstances, &root.hangarinstances);
                vec.insert(0, **shipid);
                vec
            })
            .flatten()
            .collect();
        let passive_ships: Vec<Key<ShipInstance>> = relevant_ships
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
        let ships: Vec<Key<ShipInstance>> = ships_in_active_fleets
            .into_iter()
            .chain(ships_in_node.into_iter())
            .chain(passive_ships.into_iter())
            .collect();
        (active_fleets, ships)
    }
    pub fn get_node_faction_forces(
        node: Key<Node>,
        factionid: Arc<Faction>,
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
    pub fn get_shipclass_num(nodeid: Key<Node>, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        root.shipinstances
            .iter()
            .filter(|(_, s)| s.get_node(root) == nodeid)
            .filter(|(_, s)| Arc::ptr_eq(&s.class, &shipclass))
            .count() as u64
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
        (((a_pos[0] - b_pos[0]) + (a_pos[1] - b_pos[1]) + (a_pos[2] - b_pos[2])) as f32).sqrt()
            as u64
    }
    pub fn process_factories(&mut self) {
        self.factoryinstancelist
            .iter_mut()
            .for_each(|f| f.process(self.efficiency));
    }
    pub fn process_shipyards(&mut self) {
        self.shipyardinstancelist
            .iter_mut()
            .for_each(|sy| sy.process(self.efficiency));
    }
    pub fn plan_ships(
        &mut self,
        nodeid: Key<Node>,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, ShipLocationFlavor, Arc<Faction>)> {
        self.shipyardinstancelist
            .iter_mut()
            .map(|shipyard| {
                let ship_plans = shipyard.plan_ships(self.efficiency, shipclasses);
                //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                ship_plans
                    .iter()
                    .map(|ship_plan| {
                        (
                            ship_plan.clone(),
                            ShipLocationFlavor::Node(nodeid),
                            self.allegiance.clone(),
                        )
                    })
                    // <^>>(
                    .collect::<Vec<_>>()
            })
            //we flatten the collection of vecs corresponding to individual shipyards, because we just want to create all the ships and don't care who created them
            .flatten()
            .collect::<Vec<_>>()
    }
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct System {
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
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

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct FactionID {
    pub index: usize,
}

impl FactionID {
    pub fn new_from_index(index: usize) -> Self {
        FactionID { index: index }
    }
}

#[derive(Debug, Clone)]
pub struct Faction {
    pub id: usize,
    pub visiblename: String, //faction name as shown to player
    pub description: String,
    pub visibility: bool,
    pub efficiencydefault: f32, //starting value for production facility efficiency
    pub efficiencytarget: f32, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    pub efficiencydelta: f32,  //rate at which efficiency changes
    pub battlescalar: f32,
    pub relations: HashMap<FactionID, f32>,
}

impl PartialEq for Faction {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Faction {}

impl Ord for Faction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Faction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Faction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct Resource {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub cargovol: u64, //how much space one unit of this resource takes up when transported by a cargo ship
    pub valuemult: u64, //how valuable the AI considers one unit of this resource to be
}

impl PartialEq for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Resource {}

impl Ord for Resource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Resource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Resource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

//GenericCargo is an entity which can be either a quantity of a resource or a shipinstance
#[derive(Debug, Clone)]
pub enum GenericCargo {
    Resource { id: Arc<Resource>, value: u64 },
    ShipInstance(Key<ShipInstance>),
}

impl GenericCargo {
    fn is_resource(self) -> Option<(Arc<Resource>, u64)> {
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
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum CollatedCargo {
    Resource(Arc<Resource>),
    ShipClass(Arc<ShipClass>),
}

impl CollatedCargo {
    fn get_volume(self, root: &Root) -> u64 {
        match self {
            CollatedCargo::Resource(k) => k.cargovol,
            CollatedCargo::ShipClass(k) => k.cargovol.unwrap_or(u64::MAX),
        }
    }
}

pub trait Stockpileness {
    fn get_resource_contents(&self) -> HashMap<Arc<Resource>, u64>;
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>>;
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64>;
    fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64;
    fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64;
    fn get_capacity(&self) -> u64;
    fn get_fullness(&self, root: &Root) -> u64;
    fn get_allowed(&self) -> Option<(Vec<Arc<Resource>>, Vec<Arc<ShipClass>>)>;
    fn get_resource_supply(&self, root: &Root, resourceid: Arc<Resource>) -> u64;
    fn get_resource_demand(&self, root: &Root, resourceid: Arc<Resource>) -> u64;
    fn get_shipclass_supply(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64;
    fn get_shipclass_demand(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64;
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
            .checked_div(class.clone().get_volume(root))
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
                .filter(|ship| Arc::ptr_eq(&root.shipinstances.get(**ship).unwrap().class, &k))
                .take(constrained_quantity as usize)
                .map(|ship| GenericCargo::ShipInstance(*ship))
                .collect(),
        };
        dbg!(&cargo);
        cargo.iter().for_each(|item| {
            self.remove(root, item.clone())
                .and_then(|remainder| rhs.insert(root, remainder))
                .and_then(|remainder| self.insert(root, remainder));
        })
    }
}

//this is a horrible incomprehensible nightmare that Amaryllis put me through for some reason
//okay, so, a year later, what this actually does is that it takes two individual stockpiles and allows them to function together as a single stockpile
impl<A: Stockpileness, B: Stockpileness> Stockpileness for (A, B) {
    fn get_resource_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.0
            .get_resource_contents()
            .iter()
            .chain(self.1.get_resource_contents().iter())
            .map(|(resource, value)| (resource.clone(), *value))
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
                *acc.entry(k.clone()).or_insert(0) += v;
                acc
            })
    }
    fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        self.0.get_resource_num(root, cargo.clone()) + self.1.get_resource_num(root, cargo.clone())
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        self.0.get_shipclass_num(root, cargo.clone())
            + self.1.get_shipclass_num(root, cargo.clone())
    }
    fn get_capacity(&self) -> u64 {
        self.0.get_capacity() + self.1.get_capacity()
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.0.get_fullness(root) + self.1.get_fullness(root)
    }
    fn get_allowed(&self) -> Option<(Vec<Arc<Resource>>, Vec<Arc<ShipClass>>)> {
        //self.0
        //    .get_allowed()
        //    .iter()
        //    .chain(self.1.get_allowed().iter())
        //    .collect()
        Some((Vec::new(), Vec::new()))
    }
    fn get_resource_supply(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.0.get_resource_supply(root, resource.clone())
            + self.1.get_resource_supply(root, resource.clone())
    }
    fn get_resource_demand(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.0.get_resource_demand(root, resource.clone())
            + self.1.get_resource_demand(root, resource.clone())
    }
    fn get_shipclass_supply(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        self.0.get_shipclass_supply(root, shipclass.clone())
            + self.1.get_shipclass_supply(root, shipclass.clone())
    }
    fn get_shipclass_demand(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        self.0.get_shipclass_demand(root, shipclass.clone())
            + self.1.get_shipclass_demand(root, shipclass.clone())
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
    pub resourcetype: Arc<Resource>,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for UnipotentResourceStockpile {
    fn get_resource_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((self.resourcetype.clone(), self.contents)).collect()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        HashSet::new()
    }
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        iter::once((
            CollatedCargo::Resource(self.resourcetype.clone()),
            self.contents,
        ))
        .collect()
    }
    fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resourcetype {
            self.contents
        } else {
            0
        }
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        0
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.contents * self.resourcetype.cargovol
    }
    fn get_allowed(&self) -> Option<(Vec<Arc<Resource>>, Vec<Arc<ShipClass>>)> {
        Some((vec![self.resourcetype.clone()], Vec::new()))
    }
    fn get_resource_supply(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        if resource == self.resourcetype {
            (self.contents * resource.cargovol).saturating_sub(self.target)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        if resource == self.resourcetype {
            self.target
                .saturating_sub(self.contents * resource.cargovol)
        } else {
            0
        }
    }
    fn get_shipclass_supply(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        0
    }
    fn get_shipclass_demand(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        0
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo.clone() {
            GenericCargo::Resource { id, value } => {
                let cargo_vol = id.cargovol;
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
    fn output_process(&mut self, efficiency: f32) {
        self.contents += (self.rate as f32 * efficiency) as u64;
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
    pub resource_contents: HashMap<Arc<Resource>, u64>,
    pub ship_contents: HashSet<Key<ShipInstance>>,
    pub allowed: Option<(Vec<Arc<Resource>>, Vec<ShipClassID>)>,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for PluripotentStockpile {
    fn get_resource_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.resource_contents.clone()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        self.ship_contents.clone()
    }
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        let resource_list = self
            .resource_contents
            .iter()
            .map(|(resource, value)| (CollatedCargo::Resource(resource.clone()), *value));
        let ship_list = self.ship_contents.iter().map(|key| {
            (
                CollatedCargo::ShipClass(root.shipinstances.get(*key).unwrap().class.clone()),
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
    fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        *self.resource_contents.get(&cargo).unwrap_or(&0)
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        self.ship_contents
            .iter()
            .filter(|key| root.shipinstances.get(**key).unwrap().class == cargo)
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
            .map(|(resource, value)| value * resource.cargovol)
            .sum::<u64>()
            + self
                .ship_contents
                .iter()
                .map(|shipid| {
                    root.shipinstances
                        .get(shipid.clone())
                        .unwrap()
                        .class
                        .cargovol
                        .unwrap_or(0)
                })
                .sum::<u64>()
    }
    //NOTE: Partially dummied out currently; waiting on removal of ship-carrying ability from stockpiles
    fn get_allowed(&self) -> Option<(Vec<Arc<Resource>>, Vec<Arc<ShipClass>>)> {
        match &self.allowed {
            Some((resource_allowed, shipclass_allowed)) => {
                Some((resource_allowed.clone(), Vec::new()))
            }
            None => None,
        }
    }
    //unlike other places, here in pluripotent stockpiles we don't take target into account when calculating supply
    //thus, items in pluripotent stockpiles always emit supply, even if the stockpile still wants more
    fn get_resource_supply(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.get_resource_num(root, resource.clone()) * resource.cargovol
    }
    fn get_resource_demand(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        if self
            .get_allowed()
            .unwrap_or((vec![resource.clone()], Vec::new()))
            .0
            .contains(&resource.clone())
        {
            self.target.saturating_sub(self.get_fullness(root))
        } else {
            0
        }
    }
    fn get_shipclass_supply(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        self.get_shipclass_num(root, shipclass.clone()) * shipclass.cargovol.unwrap_or(0)
    }
    fn get_shipclass_demand(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        if self
            .get_allowed()
            .unwrap_or((Vec::new(), vec![shipclass.clone()]))
            .1
            .contains(&shipclass)
        {
            self.target.saturating_sub(self.get_fullness(root))
        } else {
            0
        }
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo.clone() {
            GenericCargo::Resource { id, value } => {
                if self
                    .allowed
                    .clone()
                    .map(|(x, _)| x.contains(&id))
                    .unwrap_or(true)
                {
                    let cargo_vol = id.cargovol;
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
                let class = root.shipinstances.get(id).unwrap().class.clone();
                let classid = ShipClassID::new_from_index(class.id);
                if self
                    .allowed
                    .clone()
                    .map(|(_, x)| x.contains(&classid))
                    .unwrap_or(true)
                {
                    let cargo_vol = class.cargovol.unwrap();
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
    pub resourcetype: Arc<Resource>,
    pub contents: Arc<AtomicU64>,
    pub rate: u64,
    pub capacity: u64,
}

impl Stockpileness for SharedStockpile {
    fn get_resource_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((
            self.resourcetype.clone(),
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_ship_contents(&self) -> HashSet<Key<ShipInstance>> {
        HashSet::new()
    }
    fn collate_contents(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        iter::once((
            CollatedCargo::Resource(self.resourcetype.clone()),
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resourcetype {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        0
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self, root: &Root) -> u64 {
        self.contents.load(atomic::Ordering::SeqCst) * self.resourcetype.cargovol
    }
    fn get_allowed(&self) -> Option<(Vec<Arc<Resource>>, Vec<Arc<ShipClass>>)> {
        Some((vec![self.resourcetype.clone()], Vec::new()))
    }
    fn get_resource_supply(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        if resource == self.resourcetype {
            self.contents.load(atomic::Ordering::SeqCst) * resource.cargovol
        } else {
            0
        }
    }
    fn get_resource_demand(&self, root: &Root, resourceid: Arc<Resource>) -> u64 {
        0
    }
    fn get_shipclass_supply(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        0
    }
    fn get_shipclass_demand(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        0
    }
    fn insert(&mut self, root: &Root, cargo: GenericCargo) -> Option<GenericCargo> {
        match cargo.clone() {
            GenericCargo::Resource { id, value } => {
                let cargo_vol = id.clone().cargovol;
                if id.clone() == self.resourcetype {
                    let count_capacity = self.capacity / cargo_vol;
                    let remainder = value.saturating_sub(
                        count_capacity - self.contents.load(atomic::Ordering::SeqCst),
                    );
                    self.contents
                        .fetch_add(value - remainder, atomic::Ordering::SeqCst);
                    Some(GenericCargo::Resource {
                        id: id.clone(),
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
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub capacity: u64,                    //total volume the hangar can hold
    pub target: u64, //volume the hangar wants to hold; this is usually either equal to capacity (for carriers) or zero (for shipyard outputs)
    pub allowed: Vec<ShipClassID>, //which shipclasses this hangar can hold
    pub ideal: HashMap<ShipClassID, u64>, //how many of each ship type the hangar wants
    pub launch_volume: u64, //how much volume the hangar can launch at one time in battle
    pub launch_interval: u64, //time between launches in battle
    pub propagate: bool, //whether or not hangar generates saliences
}

impl PartialEq for HangarClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for HangarClass {}

impl Ord for HangarClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for HangarClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for HangarClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl HangarClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.ideal
            .iter()
            .map(|(shipclassid, v)| {
                root.shipclasses
                    .iter()
                    .find(|item| item.id == shipclassid.index)
                    .unwrap()
                    .get_ideal_strength(root)
                    * v
            })
            .sum()
    }
    pub fn instantiate(
        class: Arc<Self>,
        index: usize,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> HangarInstance {
        HangarInstance {
            id: Key::new_from_index(index),
            hangarclass: class.clone(),
            visibility: class.visibility,
            capacity: class.capacity,
            target: class.target,
            allowed: class
                .allowed
                .iter()
                .map(|shipclassid| {
                    (shipclasses
                        .iter()
                        .find(|shipclass| shipclass.id == shipclassid.index))
                    .unwrap()
                    .clone()
                })
                .collect(),
            ideal: class
                .ideal
                .iter()
                .map(|(shipclassid, num)| {
                    (
                        shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                        *num,
                    )
                })
                .collect(),
            launch_volume: class.launch_volume,
            launch_interval: class.launch_interval,
            propagate: class.propagate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HangarInstance {
    pub id: Key<HangarInstance>,
    pub hangarclass: Arc<HangarClass>,
    pub visibility: bool,
    pub capacity: u64,                       //total volume the hangar can hold
    pub target: u64, //volume the hangar wants to hold; this is usually either equal to capacity (for carriers) or zero (for shipyard outputs)
    pub allowed: Vec<Arc<ShipClass>>, //which shipclasses this hangar can hold
    pub ideal: HashMap<Arc<ShipClass>, u64>, //how many of each ship type the hangar wants
    pub launch_volume: u64, //how much volume the hangar can launch at one time in battle
    pub launch_interval: u64, //time between launches in battle
    pub propagate: bool,
}

impl HangarInstance {
    pub fn get_contents(&self, shipinstances: &Table<ShipInstance>) -> Vec<Key<ShipInstance>> {
        shipinstances
            .iter()
            .filter(|(_, s)| s.location == ShipLocationFlavor::Hangar(self.id))
            .map(|(shipid, _)| *shipid)
            .collect()
    }
    pub fn get_strength(&self, root: &Root, time: u64) -> u64 {
        let contents = self.get_contents(&root.shipinstances);
        let contents_strength = contents
            .iter()
            .map(|s| root.shipinstances.get(*s).unwrap().get_strength(root, time))
            .sum::<u64>() as f32;
        let contents_vol = contents
            .iter()
            .map(|s| root.shipinstances.get(*s).unwrap().class.cargovol.unwrap())
            .sum::<u64>() as f32;
        //we calculate how much of its complement the hangar can launch during a battle a certain number of seconds long
        let launch_mod = ((contents_vol / self.launch_volume as f32)
            * (time as f32 / self.launch_interval as f32))
            .clamp(0.0, 1.0);
        (contents_strength * launch_mod) as u64
    }
    pub fn get_shipclass_num(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        self.get_contents(&root.shipinstances)
            .iter()
            .filter(|s| root.shipinstances.get(**s).unwrap().class == shipclass)
            .collect::<Vec<_>>()
            .len()
            .try_into()
            .unwrap()
    }
    pub fn get_shipclass_supply(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        self.get_contents(&root.shipinstances)
            .iter()
            .filter(|s| root.shipinstances.get(**s).unwrap().class == shipclass)
            .map(|_| shipclass.cargovol.unwrap())
            .sum()
    }
    pub fn get_shipclass_demand(&self, root: &Root, shipclass: Arc<ShipClass>) -> u64 {
        let ideal_num = self.ideal.get(&shipclass).unwrap_or(&0);
        ideal_num.saturating_sub(self.get_shipclass_num(root, shipclass.clone()))
            * shipclass.cargovol.unwrap()
    }
}

fn collapse_cargo_maps(vec: &Vec<HashMap<CollatedCargo, u64>>) -> HashMap<CollatedCargo, u64> {
    vec.iter()
        .flatten()
        .fold(HashMap::new(), |mut acc, (cargo, num)| {
            *acc.entry(cargo.clone()).or_insert(0) += num;
            acc
        })
}

pub trait ResourceProcess {
    fn get_state(&self) -> FactoryState;
    fn get_resource_supply_total(&self, root: &Root, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_total(&self, root: &Root, resource: Arc<Resource>) -> u64;
    fn get_target_total(&self) -> u64;
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64>;
    fn get_output_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64;
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineClass {
    pub id: Key<EngineClass>,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub basehealth: Option<u64>,
    pub toughnessscalar: f32,
    pub inputs: Vec<UnipotentResourceStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Key<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl EngineClass {
    pub fn instantiate(&self) -> EngineInstance {
        EngineInstance {
            engineclass: self.id,
            visibility: self.visibility,
            basehealth: self.basehealth,
            health: self.basehealth,
            toughnessscalar: self.toughnessscalar,
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
    visibility: bool,
    basehealth: Option<u64>,
    health: Option<u64>,
    toughnessscalar: f32,
    inputs: Vec<UnipotentResourceStockpile>,
    forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
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
    fn get_resource_supply_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        HashMap::new()
    }
    fn get_output_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
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
                .filter(|destinationid| self.nav_check(root, location, **destinationid))
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
        destination: Key<Node>,
    ) -> Option<(u64)> {
        if (self.health != Some(0))
            && (root.turn - self.last_move_turn > self.cooldown)
            && (self.get_state() == FactoryState::Active)
            && (self.nav_check(root, location, destination))
        {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.last_move_turn = root.turn;
            Some((self.speed))
        } else {
            None
        }
    }
    //For ownership reasons, this version of process_engine does not run checks to ensure the engine is capable of performing the movement in question.
    //If used for an engine that is performing a movement, this must only be used immediately after the engine has been checked using the same movement data!
    //Otherwise the engine may perform forbidden movements.
    fn process_engine_unchecked(&mut self, turn: u64) -> Option<(u64)> {
        if (self.health != Some(0)) && (self.get_state() == FactoryState::Active) {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.last_move_turn = turn;
            Some((self.speed))
        } else {
            None
        }
    }
    fn nav_check(&self, root: &Root, location: Key<Node>, destination: Key<Node>) -> bool {
        !self
            .forbidden_nodeflavors
            .contains(&root.nodes.get(destination).unwrap().flavor)
            && root
                .edges
                .get(&(location.min(destination), destination.max(location)))
                .map(|edge| !self.forbidden_edgeflavors.contains(edge))
                .unwrap_or(false)
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
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
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
            per_engagement: self.per_engagement,
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
    per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
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
    fn get_resource_supply_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        HashMap::new()
    }
    fn get_output_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
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
    fn get_resource_supply_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.outputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_supply(root, resource.clone()))
            .sum::<u64>()
    }
    fn get_resource_demand_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resource.clone()))
            .sum::<u64>()
            + self
                .outputs
                .iter()
                .filter(|sp| sp.propagate)
                .map(|sp| sp.get_resource_demand(root, resource.clone()))
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
    fn get_output_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        self.outputs
            .iter()
            .map(|sp| sp.get_resource_num(root, cargo.clone()))
            .sum()
    }
}

impl FactoryInstance {
    //we take an active factory and update all its inputs and outputs to add or remove resources
    fn process(&mut self, location_efficiency: f32) {
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
    pub outputs: HashMap<ShipClassID, u64>,
    pub constructrate: u64,
    pub efficiency: f32,
}

impl ShipyardClass {
    pub fn instantiate(&self, shipclasses: &Vec<Arc<ShipClass>>) -> ShipyardInstance {
        ShipyardInstance {
            shipyardclass: self.id,
            visibility: self.visibility,
            inputs: self.inputs.clone(),
            outputs: self
                .outputs
                .iter()
                .map(|(shipclassid, num)| {
                    (
                        shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                        *num,
                    )
                })
                .collect(),
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
    outputs: HashMap<Arc<ShipClass>, u64>,
    constructpoints: u64,
    constructrate: u64,
    efficiency: f32,
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

    fn get_resource_supply_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        0
    }

    fn get_resource_demand_total(&self, root: &Root, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, resource.clone()))
            .sum()
    }

    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }

    fn collate_outputs(&self, root: &Root) -> HashMap<CollatedCargo, u64> {
        HashMap::new()
    }

    fn get_output_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        0
    }
}

impl ShipyardInstance {
    fn process(&mut self, location_efficiency: f32) {
        if let FactoryState::Active = self.get_state() {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.constructpoints += (self.constructrate as f32 * location_efficiency) as u64;
        }
    }

    fn try_choose_ship(&mut self, shipclasses: &Vec<Arc<ShipClass>>) -> Option<Arc<ShipClass>> {
        //we go through the list of ships the shipyard can produce, specified as its outputs, and grab the one with the highest desirability weight
        let shipclass = self
            .outputs
            .iter()
            .max_by_key(|(_, weight)| *weight)
            .unwrap()
            .0;
        let cost = shipclass.basestrength;
        //then, if the shipyard has enough points to build it, we subtract the cost and return the ship class id
        if self.constructpoints >= cost {
            self.constructpoints -= cost;
            Some(shipclass.clone())
        //otherwise we return nothing
        } else {
            None
        }
    }

    //this uses try_choose_ship to generate the list of ships the shipyard is building this turn
    fn plan_ships(
        &mut self,
        location_efficiency: f32,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<Arc<ShipClass>> {
        self.process(location_efficiency);
        (0..)
            .map_while(|_| self.try_choose_ship(shipclasses))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ShipAI {
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<ShipClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ShipLocationFlavor {
    Node(Key<Node>),
    Fleet(Key<FleetInstance>),
    Hangar(Key<HangarInstance>),
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct CargoStat {
    cargocap: u64,
    resourcecont: (Arc<Resource>, u64),
    shipcont: Vec<Key<ShipInstance>>,
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
enum CargoFlavor {
    Resource((Arc<Resource>, u64)),
    ShipInstance(Vec<Key<ShipInstance>>),
}

impl CargoFlavor {
    fn cargocapused(
        &self,
        resourcetable: &HashMap<Arc<Resource>, Resource>,
        shipinstancetable: &Table<ShipInstance>,
    ) -> u64 {
        match self {
            Self::Resource((id, n)) => resourcetable.get(id).unwrap().cargovol * n,
            Self::ShipInstance(ids) => ids
                .iter()
                .map(|&id| shipinstancetable.get(id).unwrap().class.cargovol.unwrap())
                .sum(),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ShipClassID {
    pub index: usize,
}

impl ShipClassID {
    pub fn new_from_index(index: usize) -> Self {
        ShipClassID { index: index }
    }
}

#[derive(Debug, Clone)]
pub struct ShipClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub basehull: u64,     //how many hull hitpoints this ship has by default
    pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub hangarvol: Option<u64>,
    pub cargovol: Option<u64>, //how much cargo space this ship takes up when transported by a cargo ship
    pub stockpiles: Vec<UnipotentResourceStockpile>,
    pub defaultweapons: Option<HashMap<Arc<Resource>, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<Arc<HangarClass>>,
    pub engines: Vec<Key<EngineClass>>,
    pub repairers: Vec<Key<RepairerClass>>,
    pub factoryclasslist: Vec<Key<FactoryClass>>,
    pub shipyardclasslist: Vec<Key<ShipyardClass>>,
    pub aiclass: Key<ShipAI>,
    pub defectchance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub toughnessscalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub battleescapescalar: f32, //is added to toughnessscalar in battles where this ship is on the losing side, trying to escape
    pub defectescapescalar: f32, //influences how likely it is that a ship of this class will, if it defects, escape to an enemy-held node with no engagement taking place
}

impl PartialEq for ShipClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipClass {}

impl Ord for ShipClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl ShipClass {
    fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.basestrength
            + self
                .hangars
                .iter()
                .map(|hangarclass| hangarclass.get_ideal_strength(root))
                .sum::<u64>()
    }
    //method to create a ship instance with this ship class
    fn instantiate(
        class: Arc<Self>,
        location: ShipLocationFlavor,
        faction: Arc<Faction>,
        index: usize,
        root: &mut Root,
    ) -> ShipInstance {
        ShipInstance {
            id: Key::new_from_index(index),
            visiblename: uuid::Uuid::new_v4().to_string(),
            class: class.clone(),
            hull: class.basehull,
            visibility: class.visibility,
            stockpiles: class.stockpiles.clone(),
            efficiency: 1.0,
            hangars: class
                .hangars
                .iter()
                .map(|hangarclass| root.create_hangar(hangarclass.clone()))
                .collect(),
            engines: class
                .engines
                .iter()
                .map(|classid| root.engineclasses.get(*classid).unwrap().instantiate())
                .collect(),
            movement_left: u64::MAX,
            repairers: class
                .repairers
                .iter()
                .map(|classid| root.repairerclasses.get(*classid).unwrap().instantiate())
                .collect(),
            factoryinstancelist: class
                .factoryclasslist
                .iter()
                .map(|classid| root.factoryclasses.get(*classid).unwrap().instantiate())
                .collect(),
            shipyardinstancelist: class
                .shipyardclasslist
                .iter()
                .map(|classid| {
                    root.shipyardclasses
                        .get(*classid)
                        .unwrap()
                        .instantiate(&root.shipclasses)
                })
                .collect(),
            location,
            allegiance: faction,
            objectives: Vec::new(),
            aiclass: class.aiclass,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShipInstance {
    pub id: Key<ShipInstance>,
    pub visiblename: String,
    pub class: Arc<ShipClass>, //which class of ship this is
    pub hull: u64,             //how many hitpoints the ship has
    pub visibility: bool,
    pub stockpiles: Vec<UnipotentResourceStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<Key<HangarInstance>>,
    pub engines: Vec<EngineInstance>,
    pub movement_left: u64, //starts at one trillion each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<RepairerInstance>,
    pub factoryinstancelist: Vec<FactoryInstance>,
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub location: ShipLocationFlavor, //where the ship is -- a node if it's unaffiliated, a fleet if it's in one
    pub allegiance: Arc<Faction>,     //which faction this ship belongs to
    pub objectives: Vec<ObjectiveFlavor>,
    pub aiclass: Key<ShipAI>,
}

impl ShipInstance {
    pub fn get_daughters(
        &self,
        shipinstances: &Table<ShipInstance>,
        hangarinstances: &Table<HangarInstance>,
    ) -> Vec<Key<ShipInstance>> {
        self.hangars
            .iter()
            .map(|h| {
                hangarinstances
                    .get(*h)
                    .unwrap()
                    .get_contents(shipinstances)
                    .iter()
                    .map(|s| {
                        let mut vec = shipinstances
                            .get(*s)
                            .unwrap()
                            .get_daughters(shipinstances, hangarinstances);
                        vec.insert(0, *s);
                        vec
                    })
                    .collect::<Vec<Vec<Key<ShipInstance>>>>()
            })
            .flatten()
            .flatten()
            .collect()
    }
    pub fn get_daughters_recursive(&self, root: &Root) -> Vec<Key<ShipInstance>> {
        self.hangars
            .iter()
            .map(|h| {
                root.hangarinstances
                    .get(*h)
                    .unwrap()
                    .get_contents(&root.shipinstances)
                    .iter()
                    .map(|s| {
                        let mut granddaughterids = root
                            .shipinstances
                            .get(*s)
                            .unwrap()
                            .get_daughters(&root.shipinstances, &root.hangarinstances);
                        granddaughterids.insert(0, *s);
                        granddaughterids
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
                let (active, passive): (Vec<Key<ShipInstance>>, Vec<Key<ShipInstance>>) = root
                    .hangarinstances
                    .get(*h)
                    .unwrap()
                    .get_contents(&root.shipinstances)
                    .iter()
                    .partition(|s| {
                        root.shipinstances
                            .get(**s)
                            .unwrap()
                            .nav_check(root, vec![destination])
                    });
                let vec1 = active
                    .iter()
                    .map(|s| {
                        let mut vec1 = root
                            .shipinstances
                            .get(*s)
                            .unwrap()
                            .get_daughters(&root.shipinstances, &root.hangarinstances);
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
    pub fn change_allegiance(&mut self, root: &mut Root, new_faction: Arc<Faction>) {
        self.allegiance = new_faction.clone();
        let all_daughterids = self.get_daughters_recursive(root);
        for daughterid in all_daughterids {
            root.shipinstances.get_mut(daughterid).unwrap().allegiance = new_faction.clone()
        }
    }
    pub fn kill(
        &self,
        shipinstances: &mut Table<ShipInstance>,
        hangarinstances: &Table<HangarInstance>,
    ) {
        let mut ships_to_kill = shipinstances
            .get(self.id)
            .unwrap()
            .get_daughters(shipinstances, hangarinstances);
        ships_to_kill.insert(0, self.id);
        ships_to_kill.iter().for_each(|s| {
            shipinstances.get_mut(*s).unwrap().hull = 0;
        });
    }
    //NOTE: Dummied out until characters exist.
    pub fn get_character_strength_scalar(&self, root: &Root) -> f32 {
        1.0
    }
    pub fn get_strength(&self, root: &Root, time: u64) -> u64 {
        let daughter_strength = self
            .hangars
            .iter()
            .map(|h| {
                root.hangarinstances
                    .get(*h)
                    .unwrap()
                    .get_strength(root, time)
            })
            .sum::<u64>();
        let objective_strength: f32 = self
            .objectives
            .iter()
            .map(|of| of.get_scalars().strengthscalar)
            .product();
        (self.class.basestrength as f32
            * (self.hull as f32 / self.class.basehull as f32)
            * self.get_character_strength_scalar(root)
            * objective_strength) as u64
            + daughter_strength
    }
    //NOTE: Dummied out until morale system exists.
    pub fn get_morale_scalar(&self, root: &Root) -> f32 {
        1.0
    }
    //Checks whether the shipinstance will defect this turn; if it will, makes the ship defect and returns the node the ship is in afterward
    //so that that node can be checked for engagements
    pub fn try_defect(&mut self, root: &Root) -> Option<Key<Node>> {
        let location = self.get_node(root);
        //NOTE: Test whether this gives us numbers that make sense.
        let local_threat_ratio: f32 = self
            .class
            .defectchance
            .iter()
            .map(|(faction, _)| {
                root.globalsalience.factionsalience[self.allegiance.id][faction.id][location.index]
                    [0]
            })
            .sum();
        //if defectchance only contains data for one faction
        //then either it's the faction it's currently part of, in which case we have no data on its chance of joining other factions
        //or it's not, in which case we have no data on its chance of defecting from its current faction
        //so either way we treat it as incapable of defecting
        let defect_probability = if self.class.defectchance.len() > 1 {
            ((local_threat_ratio * self.class.defectchance
                .get(&self.allegiance)
                .unwrap_or(&(0.0, 0.0))
                .0)
                //NOTE: I *think* that clamp will resolve things properly if we end up dividing by zero -- it'll set the probability to 1 -- but it's hard to be sure
                /self.get_morale_scalar(root))
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut rng = thread_rng();
        let defects = rng.gen_bool(defect_probability as f64);
        if defects {
            let new_faction_probabilities: Vec<(Arc<Faction>, f32)> = self
                .class
                .defectchance
                .iter()
                .map(|(faction, (_, defect_to))| {
                    (
                        faction.clone(),
                        (defect_to
                            * root.globalsalience.factionsalience[faction.id][faction.id]
                                [location.index][0]),
                    )
                })
                .collect();
            let new_faction: Arc<Faction> = new_faction_probabilities
                .choose_weighted(&mut rng, |(_, prob)| prob.clone())
                .unwrap()
                .0
                .clone();
            self.allegiance = new_faction.clone();
            //NOTE: This should take more things into account probably
            let escapes = rng.gen_bool(self.class.defectescapescalar.clamp(0.0, 1.0) as f64);
            if escapes {
                let destinations_option =
                    self.get_checked_destinations(root, root.neighbors.get(&location).unwrap());
                match destinations_option {
                    Some(destinations) => {
                        let destination = destinations
                            .iter()
                            .max_by_key(|nodeid| {
                                root.globalsalience.factionsalience[new_faction.id][new_faction.id]
                                    [nodeid.index][0] as i64
                            })
                            .unwrap();
                        self.traverse(root, *destination);
                        Some(*destination)
                    }
                    None => {
                        self.location = ShipLocationFlavor::Node(location);
                        Some(location)
                    }
                }
            } else {
                self.location = ShipLocationFlavor::Node(location);
                Some(location)
            }
        } else {
            None
        }
    }
    pub fn get_checked_destinations(
        &self,
        root: &Root,
        destinations: &Vec<Key<Node>>,
    ) -> Option<Vec<Key<Node>>> {
        let location = self.get_node(root);
        if self.movement_left > 0 {
            if let Some((viable, speed)) = self
                .engines
                .iter()
                .find_map(|e| e.check_engine(root, location, destinations))
            {
                Some(viable)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn process_engines(&mut self, root: &Root, destination: Key<Node>) {
        let location = self.get_node(root);
        if self.movement_left > 0 {
            if let Some(speed) = self
                .engines
                .iter_mut()
                .find_map(|e| e.process_engine(root, location, destination))
            {
                self.movement_left.saturating_sub(u64::MAX / speed);
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
    //For ownership reasons, this version of process_engines does not run checks to ensure the ship's engines are capable of performing the movement in question.
    //If used for an ship that is performing a movement, this must only be used immediately after the ship has been checked using the same movement data!
    //Otherwise the ship may perform forbidden movements.
    pub fn process_engines_unchecked(&mut self, turn: u64) {
        if self.movement_left > 0 {
            if let Some(speed) = self
                .engines
                .iter_mut()
                .find_map(|e| e.process_engine_unchecked(turn))
            {
                self.movement_left.saturating_sub(u64::MAX / speed);
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
    pub fn reset_movement(&mut self) {
        self.movement_left = u64::MAX;
    }
    pub fn repair_turn(&mut self) {
        if self.hull < self.class.basehull || self.engines.iter().any(|e| e.health < e.basehealth) {
            self.repairers
                .iter()
                .filter(|rp| !rp.per_engagement)
                .filter(|rp| rp.get_state() == FactoryState::Active)
                .for_each(|rp| {
                    self.hull = (self.hull as i64
                        + rp.repair_points
                        + (self.class.basehull as f32 * rp.repair_factor) as i64)
                        .clamp(0, self.class.basehull as i64)
                        as u64;
                    self.engines
                        .iter_mut()
                        .filter(|e| e.health.is_some())
                        .for_each(|e| {
                            (e.health.unwrap() as i64
                                + rp.engine_repair_points
                                + (e.basehealth.unwrap() as f32 * rp.engine_repair_factor) as i64)
                                .clamp(0, e.basehealth.unwrap() as i64)
                                as u64;
                        })
                })
        }
    }
    pub fn repair_engagement(&mut self) {
        if self.hull < self.class.basehull || self.engines.iter().any(|e| e.health < e.basehealth) {
            self.repairers
                .iter()
                .filter(|rp| rp.per_engagement)
                .filter(|rp| rp.get_state() == FactoryState::Active)
                .for_each(|rp| {
                    self.hull = (self.hull as i64
                        + rp.repair_points
                        + (self.class.basehull as f32 * rp.repair_factor) as i64)
                        .clamp(0, self.class.basehull as i64)
                        as u64;
                    self.engines
                        .iter_mut()
                        .filter(|e| e.health.is_some())
                        .for_each(|e| {
                            (e.health.unwrap() as i64
                                + rp.engine_repair_points
                                + (e.basehealth.unwrap() as f32 * rp.engine_repair_factor) as i64)
                                .clamp(0, e.basehealth.unwrap() as i64)
                                as u64;
                        })
                })
        }
    }
    pub fn process_factories(&mut self) {
        self.factoryinstancelist
            .iter_mut()
            .for_each(|f| f.process(self.efficiency));
    }
    pub fn process_shipyards(&mut self) {
        self.shipyardinstancelist
            .iter_mut()
            .for_each(|sy| sy.process(self.efficiency));
    }
    pub fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_resource_num(root, cargo.clone()))
            .sum::<u64>()
            + self
                .factoryinstancelist
                .iter()
                .map(|f| f.get_output_resource_num(root, cargo.clone()))
                .sum::<u64>()
    }
    pub fn get_resource_supply(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        self.stockpiles
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_supply(root, cargo.clone()))
            .sum::<u64>()
            + self
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_supply_total(root, cargo.clone()))
                .sum::<u64>()
    }
    pub fn get_resource_demand(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        self.stockpiles
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(root, cargo.clone()))
            .sum::<u64>()
            + self
                .engines
                .iter()
                .map(|e| e.get_resource_demand_total(root, cargo.clone()))
                .sum::<u64>()
            + self
                .repairers
                .iter()
                .map(|r| r.get_resource_demand_total(root, cargo.clone()))
                .sum::<u64>()
            + self
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_demand_total(root, cargo.clone()))
                .sum::<u64>()
            + self
                .shipyardinstancelist
                .iter()
                .map(|s| s.get_resource_demand_total(root, cargo.clone()))
                .sum::<u64>()
    }
    pub fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_num(root, cargo.clone()))
            .sum::<u64>()
            + self
                .hangars
                .iter()
                .map(|h| {
                    root.hangarinstances
                        .get(*h)
                        .unwrap()
                        .get_shipclass_num(root, cargo.clone())
                })
                .sum::<u64>()
    }
    pub fn get_shipclass_supply(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_supply(root, cargo.clone()))
            .sum::<u64>()
            + self
                .hangars
                .iter()
                .map(|h| {
                    root.hangarinstances
                        .get(*h)
                        .unwrap()
                        .get_shipclass_supply(root, cargo.clone())
                })
                .sum::<u64>()
            + if self.class == cargo {
                self.class.cargovol.unwrap()
            } else {
                0
            }
    }
    pub fn get_shipclass_demand(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        self.stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_demand(root, cargo.clone()))
            .sum::<u64>()
            + self
                .hangars
                .iter()
                .map(|h| {
                    root.hangarinstances
                        .get(*h)
                        .unwrap()
                        .get_shipclass_demand(root, cargo.clone())
                })
                .sum::<u64>()
    }
    pub fn get_resource_demand_ratio(&self, root: &Root, resource: Arc<Resource>) -> f32 {
        let demand_total = self
            .stockpiles
            .iter()
            .map(|sp| sp.get_resource_demand(root, resource.clone()))
            .sum::<u64>() as f32;
        let target_total = self.stockpiles.iter().map(|sp| sp.target).sum::<u64>() as f32;
        assert!(demand_total < target_total);
        demand_total / target_total
    }
    pub fn get_shipclass_demand_ratio(&self, root: &Root, shipclass: Arc<ShipClass>) -> f32 {
        let demand_total = self
            .stockpiles
            .iter()
            .map(|sp| sp.get_shipclass_demand(root, shipclass.clone()))
            .sum::<u64>() as f32;
        let target_total = self.stockpiles.iter().map(|sp| sp.target).sum::<u64>() as f32;
        assert!(demand_total < target_total);
        demand_total / target_total
    }
    pub fn plan_ships(
        &mut self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, ShipLocationFlavor, Arc<Faction>)> {
        self.shipyardinstancelist
            .iter_mut()
            .map(|shipyard| {
                let ship_plans = shipyard.plan_ships(self.efficiency, shipclasses);
                //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                ship_plans
                    .iter()
                    .map(|ship_plan| (ship_plan.clone(), self.location, self.allegiance.clone()))
                    // <^>>(
                    .collect::<Vec<_>>()
            })
            //we flatten the collection of vecs corresponding to individual shipyards, because we just want to create all the ships and don't care who created them
            .flatten()
            .collect::<Vec<_>>()
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
            ShipLocationFlavor::Hangar(k) => root
                .shipinstances
                .iter()
                .find(|(_, s)| s.hangars.iter().any(|h| *h == k))
                .map(|(_, s)| s)
                .unwrap()
                .is_in_fleet(root),
        }
    }
    pub fn get_fleet(&self, root: &Root) -> Option<Key<FleetInstance>> {
        match self.location {
            ShipLocationFlavor::Node(_) => None,
            ShipLocationFlavor::Fleet(k) => Some(k),
            ShipLocationFlavor::Hangar(k) => root
                .shipinstances
                .iter()
                .find(|(_, s)| s.hangars.iter().any(|h| *h == k))
                .map(|(_, s)| s)
                .unwrap()
                .get_fleet(root),
        }
    }
    pub fn get_carrier(&self, root: &Root) -> Key<ShipInstance> {
        match root.shipinstances.get(self.id).unwrap().location {
            ShipLocationFlavor::Node(_) => self.id,
            ShipLocationFlavor::Fleet(_) => self.id,
            ShipLocationFlavor::Hangar(k) => self.get_carrier(root),
        }
    }
    //determines which node the ship is in
    //a ship can be in a number of places which aren't directly in a node, but all of them cash out to a node eventually
    pub fn get_node(&self, root: &Root) -> Key<Node> {
        match self.location {
            ShipLocationFlavor::Node(id) => id,
            ShipLocationFlavor::Fleet(id) => root.fleetinstances.get(id).unwrap().location,
            ShipLocationFlavor::Hangar(id) => root
                .shipinstances
                .iter()
                .find(|(_, s)| s.hangars.iter().any(|h| *h == id))
                .map(|(_, s)| s)
                .unwrap()
                .get_node(root),
        }
    }
    pub fn nav_check(&self, root: &Root, destinations: Vec<Key<Node>>) -> bool {
        self.get_checked_destinations(root, &destinations).is_some()
    }
    pub fn traverse(&mut self, root: &Root, destination: Key<Node>) -> Option<Key<Node>> {
        self.process_engines(root, destination);
        self.location = ShipLocationFlavor::Node(destination);
        Some(destination)
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
                let ai = root.shipais.get(self.aiclass).unwrap();
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                let resource_demand_value: f32 = ai
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        let demand = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][nodeid.index][0];
                        let supply = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][nodeid.index][1];
                        //let cargo = self.stockpiles.iter().map(|x|)
                        (demand - supply)
                            * (self.get_resource_num(root, resource.clone()) as f32
                                * resource.cargovol as f32)
                            * scalar
                    })
                    .sum();
                let resource_supply_value: f32 = ai
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not position.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        let demand = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][location.index][0];
                        let supply = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][nodeid.index][1];
                        supply
                            * demand
                            * self.get_resource_demand_ratio(root, resource.clone())
                            * scalar
                    })
                    .sum();
                let shipclass_demand_value: f32 = ai
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        let attractive_shipclass = root
                            .shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap();
                        let demand = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][nodeid.index][0];
                        let supply = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][nodeid.index][1];
                        (demand - supply)
                            * (self.get_shipclass_num(root, attractive_shipclass.clone()) as f32
                                * attractive_shipclass.cargovol.unwrap_or(0) as f32)
                            * scalar
                    })
                    .sum();
                //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
                let shipclass_supply_value: f32 = ai
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not location.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        let attractive_shipclass = root
                            .shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap();
                        let demand = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][location.index][0];
                        let supply = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][nodeid.index][1];
                        supply
                            * demand
                            * self.get_shipclass_demand_ratio(root, attractive_shipclass.clone())
                            * scalar
                    })
                    .sum();
                //this checks how much demand there is in the node for ships of the subject ship's class
                let ship_value_specific: f32 = root.globalsalience.shipclasssalience
                    [self.allegiance.id][self.class.id][nodeid.index][0]
                    * ai.ship_attract_specific;
                //oh, THIS is why we needed the placeholder ship class
                //this checks how much demand there is in the node for ships in general
                let ship_value_generic: f32 = root.globalsalience.shipclasssalience
                    [self.allegiance.id][0][nodeid.index][0]
                    * ai.ship_attract_generic;

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
    pub fn maneuver(&mut self, root: &Root) -> Option<Key<Node>> {
        let location = self.get_node(root);
        let neighbors = root.neighbors.get(&location).unwrap();
        if let Some(destinations) = self.get_checked_destinations(root, &neighbors) {
            let destination_option = self.navigate(root, &destinations);
            match destination_option {
                Some(destination) => {
                    self.traverse(root, destination);
                }
                None => {}
            }
            destination_option
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FleetClass {
    pub id: Key<FleetClass>,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub strengthmod: (f32, u64),
    pub fleetconfig: HashMap<Arc<ShipClass>, u64>,
    pub defectchance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub defectescapescalar: f32,
    pub navthreshold: f32,
    pub disbandthreshold: f32,
}

impl FleetClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.fleetconfig
            .iter()
            .map(|(shipclass, v)| shipclass.get_ideal_strength(root) * v)
            .sum()
    }
    pub fn instantiate(
        &self,
        location: Key<Node>,
        faction: Arc<Faction>,
        index: usize,
        root: &Root,
    ) -> FleetInstance {
        FleetInstance {
            id: Key::new_from_index(index),
            visiblename: uuid::Uuid::new_v4().to_string(),
            fleetclass: self.id,
            visibility: self.visibility,
            idealstrength: self.get_ideal_strength(root),
            location: location,
            allegiance: faction,
            objectives: Vec::new(),
            phantom: true,
            fleetconfig: self.fleetconfig.clone(),
            defectchance: self.defectchance.clone(),
            defectescapescalar: self.defectescapescalar,
            navthreshold: self.navthreshold,
            disbandthreshold: self.disbandthreshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FleetInstance {
    id: Key<FleetInstance>,
    visiblename: String,
    fleetclass: Key<FleetClass>,
    visibility: bool,
    idealstrength: u64,
    location: Key<Node>,
    allegiance: Arc<Faction>,
    objectives: Vec<ObjectiveFlavor>,
    phantom: bool,
    fleetconfig: HashMap<Arc<ShipClass>, u64>,
    defectchance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    defectescapescalar: f32,
    navthreshold: f32,
    disbandthreshold: f32,
}

impl FleetInstance {
    pub fn get_daughters(&self, root: &Root) -> Vec<Key<ShipInstance>> {
        root.shipinstances
            .iter()
            .filter(|(_, s)| s.get_fleet(root) == Some(self.id))
            .map(|(id, _)| *id)
            .collect()
    }
    pub fn get_strength(&self, root: &Root, time: u64) -> u64 {
        let (factor, additive) = root.fleetclasses.get(self.fleetclass).unwrap().strengthmod;
        let sum = self
            .get_daughters(root)
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
    pub fn get_morale_scalar(&self, root: &Root) -> f32 {
        self.get_daughters(root)
            .iter()
            .map(|shipid| {
                root.shipinstances
                    .get(*shipid)
                    .unwrap()
                    .get_morale_scalar(root)
            })
            .product()
    }
    pub fn change_allegiance(&mut self, root: &mut Root, new_faction: Arc<Faction>) {
        self.allegiance = new_faction.clone();
        let all_daughterids: Vec<Key<ShipInstance>> = self
            .get_daughters(root)
            .iter()
            .map(|daughterid| {
                let granddaughterids = root
                    .shipinstances
                    .get(*daughterid)
                    .unwrap()
                    .get_daughters_recursive(root);
                vec![vec![*daughterid], granddaughterids]
            })
            .flatten()
            .flatten()
            .collect();
        for daughterid in all_daughterids {
            root.shipinstances.get_mut(daughterid).unwrap().allegiance = new_faction.clone()
        }
    }
    //Checks whether the fleetinstance will defect this turn; if it will, makes the ship defect and returns the node the ship is in afterward
    //so that that node can be checked for engagements
    pub fn try_defect(&mut self, root: &mut Root) -> Option<Vec<Key<Node>>> {
        //if defectchance only contains data for one faction
        //then either it's the faction it's currently part of, in which case we have no data on its chance of joining other factions
        //or it's not, in which case we have no data on its chance of defecting from its current faction
        //so either way we treat it as incapable of defecting
        //NOTE: Test whether this gives us numbers that make sense.
        let local_threat_ratio: f32 = self
            .defectchance
            .iter()
            .map(|(faction, _)| {
                root.globalsalience.factionsalience[self.allegiance.id][faction.id]
                    [self.location.index][0]
            })
            .sum();
        let defect_probability = if self.defectchance.len() > 1 {
            ((local_threat_ratio * self.defectchance
                .get(&self.allegiance)
                .unwrap_or(&(0.0, 0.0))
                .0)
                //NOTE: I *think* that clamp will resolve things properly if we end up dividing by zero -- it'll set the probability to 1 -- but it's hard to be sure
                /self.get_morale_scalar(root))
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut rng = thread_rng();
        let defects = rng.gen_bool(defect_probability as f64);
        if defects {
            let new_faction_probabilities: Vec<(Arc<Faction>, f32)> = self
                .defectchance
                .iter()
                .map(|(faction, (_, defect_to))| {
                    (
                        faction.clone(),
                        (defect_to
                            * root.globalsalience.factionsalience[faction.id][faction.id]
                                [self.location.index][0]),
                    )
                })
                .collect();
            let new_faction: Arc<Faction> = new_faction_probabilities
                .choose_weighted(&mut rng, |(_, prob)| prob.clone())
                .unwrap()
                .0
                .clone();
            self.change_allegiance(root, new_faction.clone());
            let escapes = rng.gen_bool(self.defectescapescalar.clamp(0.0, 1.0) as f64);
            if escapes {
                let neighbors = root.neighbors.get(&self.location).unwrap().clone();
                let destinations_option = self.get_checked_destinations(root, neighbors);
                match destinations_option {
                    Some(destinations) => {
                        let destination = destinations
                            .iter()
                            .max_by_key(|nodeid| {
                                root.globalsalience.factionsalience[new_faction.id][new_faction.id]
                                    [nodeid.index][0] as i64
                            })
                            .unwrap();
                        self.traverse(root, *destination);
                        Some(vec![self.location, *destination])
                    }
                    None => Some(vec![self.location]),
                }
            } else {
                Some(vec![self.location])
            }
        } else {
            None
        }
    }
    pub fn get_ai(&self, root: &Root) -> ShipAI {
        self.get_daughters(root).iter().fold(
            ShipAI {
                ship_attract_specific: 1.0,
                ship_attract_generic: 1.0,
                ship_cargo_attract: HashMap::new(),
                resource_attract: HashMap::new(),
            },
            |mut acc, shipid| {
                let sub_ai = root
                    .shipais
                    .get(root.shipinstances.get(*shipid).unwrap().aiclass)
                    .unwrap();
                acc.ship_attract_specific *= sub_ai.ship_attract_specific;
                acc.ship_attract_generic *= sub_ai.ship_attract_generic;
                sub_ai.ship_cargo_attract.iter().for_each(|(scid, num)| {
                    *acc.ship_cargo_attract.entry(*scid).or_insert(1.0) *= num;
                });
                sub_ai.ship_cargo_attract.iter().for_each(|(rid, num)| {
                    *acc.ship_cargo_attract.entry(*rid).or_insert(1.0) *= num;
                });
                acc
            },
        )
    }
    pub fn get_resource_num(&self, root: &Root, cargo: Arc<Resource>) -> u64 {
        self.get_daughters(root)
            .iter()
            .map(|s| {
                root.shipinstances
                    .get(*s)
                    .unwrap()
                    .get_resource_num(root, cargo.clone())
            })
            .sum()
    }
    pub fn get_resource_demand_ratio(&self, root: &Root, resource: Arc<Resource>) -> f32 {
        let daughters = self.get_daughters(root);
        daughters
            .iter()
            .map(|s| {
                root.shipinstances
                    .get(*s)
                    .unwrap()
                    .get_resource_demand_ratio(root, resource.clone())
            })
            .sum::<f32>()
            / daughters.len() as f32
    }
    pub fn get_shipclass_num(&self, root: &Root, cargo: Arc<ShipClass>) -> u64 {
        self.get_daughters(root)
            .iter()
            .map(|s| {
                root.shipinstances
                    .get(*s)
                    .unwrap()
                    .get_shipclass_num(root, cargo.clone())
            })
            .sum()
    }
    pub fn get_shipclass_demand_ratio(&self, root: &Root, shipclass: Arc<ShipClass>) -> f32 {
        let daughters = self.get_daughters(root);
        daughters
            .iter()
            .map(|s| {
                root.shipinstances
                    .get(*s)
                    .unwrap()
                    .get_shipclass_demand_ratio(root, shipclass.clone())
            })
            .sum::<f32>()
            / daughters.len() as f32
    }
    pub fn expel(&self, shipinstances: &mut Table<ShipInstance>, ships: Vec<Key<ShipInstance>>) {
        ships.iter().for_each(|s| {
            shipinstances.get_mut(*s).unwrap().location = ShipLocationFlavor::Node(self.location)
        });
    }
    pub fn disband(&self, root: &mut Root) {
        let daughters = self.get_daughters(root);
        for shipid in daughters {
            root.shipinstances.get_mut(shipid).unwrap().location =
                ShipLocationFlavor::Node(root.fleetinstances.get(self.id).unwrap().location);
        }
    }
    pub fn get_node(&self, _root: &Root) -> Key<Node> {
        self.location
    }
    pub fn process_engines(&self, root: &mut Root, destination: Key<Node>) {
        let turn = root.turn;
        let daughterids = self.get_daughters(root);
        for daughterid in &daughterids {
            let daughter = root.shipinstances.get(*daughterid).unwrap();
            assert!(daughter.nav_check(root, vec![destination]));
        }
        for daughterid in daughterids {
            let daughter = root.shipinstances.get_mut(daughterid).unwrap();
            daughter.process_engines_unchecked(turn)
        }
    }
    pub fn nav_check(&self, root: &Root, destination: Key<Node>) -> Option<Vec<Key<ShipInstance>>> {
        let daughters = self.get_daughters(root);
        let (passed_ships, failed_ships): (Vec<Key<ShipInstance>>, Vec<Key<ShipInstance>>) =
            daughters.iter().partition(|(id)| {
                //NOTE: fucky redundancy
                root.shipinstances
                    .get(**id)
                    .unwrap()
                    .nav_check(root, vec![destination])
            });
        //we see what fraction of the fleet's strength is able to make the jump
        //by checking strength of passed ships, and then all daughters
        //we don't just call get_strength on the fleet itself
        //if we did, the fleet's strength modifiers would be counted only toward its total
        if passed_ships
            .iter()
            .map(|id| {
                root.shipinstances
                    .get(*id)
                    .unwrap()
                    .get_strength(root, root.config.battlescalars.avg_duration)
                    as f32
            })
            .sum::<f32>()
            / daughters
                .iter()
                .map(|daughterid| {
                    root.shipinstances
                        .get(*daughterid)
                        .unwrap()
                        .get_strength(root, root.config.battlescalars.avg_duration)
                        as f32
                })
                .sum::<f32>()
            > self.navthreshold
        {
            Some(failed_ships)
        } else {
            None
        }
    }
    pub fn get_checked_destinations(
        &self,
        root: &Root,
        destinations: Vec<Key<Node>>,
    ) -> Option<Vec<Key<Node>>> {
        let viable: Vec<_> = destinations
            .iter()
            .filter(|nodeid| self.nav_check(root, **nodeid).is_some())
            .copied()
            .collect();
        if !viable.is_empty() {
            Some(viable)
        } else {
            None
        }
    }
    pub fn traverse(&mut self, root: &mut Root, destination: Key<Node>) -> Option<Key<Node>> {
        if let Some(left_behind) = self.nav_check(root, destination) {
            self.expel(&mut root.shipinstances, left_behind);
            self.process_engines(root, destination);
            self.location = destination;
            Some(destination)
        } else {
            None
        }
    }
    pub fn navigate(&self, root: &Root, destinations: &Vec<Key<Node>>) -> Option<Key<Node>> {
        //we iterate over the destinations to determine which neighbor is most desirable
        destinations
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .max_by_key(|nodeid| {
                let ai = self.get_ai(root);
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                let resource_demand_value: f32 = ai
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        let demand = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][nodeid.index][0];
                        let supply = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][nodeid.index][1];
                        (demand - supply)
                            * (self.get_resource_num(root, resource.clone()) as f32
                                * resource.cargovol as f32)
                            * scalar
                    })
                    .sum();
                let resource_supply_value: f32 = ai
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not position.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        let demand = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][self.location.index][0];
                        let supply = root.globalsalience.resourcesalience[self.allegiance.id]
                            [resource.id][nodeid.index][1];
                        supply
                            * demand
                            * self.get_resource_demand_ratio(root, resource.clone())
                            * scalar
                    })
                    .sum();
                let shipclass_demand_value: f32 = ai
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        let attractive_shipclass = root
                            .shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap();
                        let demand = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][nodeid.index][0];
                        let supply = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][nodeid.index][1];
                        (demand - supply)
                            * (self.get_shipclass_num(root, attractive_shipclass.clone()) as f32
                                * attractive_shipclass.cargovol.unwrap_or(0) as f32)
                            * scalar
                    })
                    .sum();
                //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
                let shipclass_supply_value: f32 = ai
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        let attractive_shipclass = root
                            .shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap();
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not location.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        //NOTE: We're indexing into a vec by shipclass id here. Make sure that works properly.
                        let demand = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][self.location.index][0];
                        let supply = root.globalsalience.shipclasssalience[self.allegiance.id]
                            [shipclassid.index][nodeid.index][1];
                        supply
                            * demand
                            * self.get_shipclass_demand_ratio(root, attractive_shipclass.clone())
                            * scalar
                    })
                    .sum();
                //NOTE: Here in the fleet implementation, I have removed the component that determines how much demand there is for ships of this ship's class.
                //In theory we could implement this by adding up the demand for the ships of the daughters' various classes, scaled according to their individual
                //ship_attract_specifics, but right now I'm not going to bother.

                //this checks how much demand there is in the node for ships in general
                let ship_value_generic: f32 = root.globalsalience.shipclasssalience
                    [self.allegiance.id][0][nodeid.index][0]
                    * ai.ship_attract_generic;

                NotNan::new(
                    resource_demand_value
                        + resource_supply_value
                        + shipclass_demand_value
                        + shipclass_supply_value
                        //+ ship_value_specific
                        + ship_value_generic,
                )
                .unwrap()
            })
            .copied()
        //if this doesn't work for some reason, we return None
    }
    pub fn maneuver(&mut self, root: &mut Root) -> Option<Key<Node>> {
        let neighbors = root.neighbors.get(&self.location).unwrap().clone();
        if let Some(destinations) = self.get_checked_destinations(root, neighbors) {
            let destination_option = self.navigate(root, &destinations);
            match destination_option {
                Some(destination) => {
                    self.traverse(root, destination);
                }
                None => {}
            }
            destination_option
        } else {
            None
        }
    }
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
    battleescapescalar: f32,
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
pub struct EngagementPrep {
    turn: u64,
    attackers: HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    defenders: HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    attacker_reinforcements:
        HashMap<Arc<Faction>, Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>>,
    defender_reinforcements:
        HashMap<Arc<Faction>, Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>>,
    location: Key<Node>,
    aggressor: Arc<Faction>,
}

impl EngagementPrep {
    pub fn engagement_prep(root: &Root, location: Key<Node>, aggressor: Arc<Faction>) -> Self {
        let belligerents = Node::get_node_forces(location, root);

        //at present there can be only one attacker faction
        //we figure out which of the belligerents is the aggressor
        //then get the attacker's fleets and ships
        let attackers: HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)> =
            belligerents
                .iter()
                .filter(|(faction, _)| *faction.clone() == aggressor)
                .map(|(faction, (fs, ss))| {
                    (
                        faction.clone(),
                        (
                            fs.clone(),
                            ss.iter().map(|s| *s).collect::<Vec<Key<ShipInstance>>>(),
                        ),
                    )
                })
                .collect();

        //we do the same for defenders, the only difference being that there can be multiple defender factions
        //we check whether each faction whose assets are currently occupying the location node is at war with the attacker
        let defenders: HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)> =
            belligerents
                .iter()
                .filter(|(faction, _)| *faction.clone() != aggressor.clone())
                .filter(|(faction, _)| {
                    root.wars.contains(&(
                        faction.clone().min(&aggressor.clone()).clone(),
                        aggressor.clone().max(faction.clone().clone()),
                    ))
                })
                .map(|(faction, (fs, ss))| {
                    (
                        faction.clone(),
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
            Arc<Faction>,
            Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = attackers
            .iter()
            .map(|(faction, _)| {
                (
                    faction.clone(),
                    root.neighbors
                        .get(&location)
                        .unwrap()
                        .iter()
                        .map(|n| {
                            (
                                Node::get_distance(*n, location, root),
                                Node::get_node_faction_reinforcements(
                                    *n,
                                    location,
                                    faction.clone(),
                                    root,
                                ),
                            )
                        })
                        .collect(),
                )
            })
            .collect();

        let defender_reinforcements: HashMap<
            Arc<Faction>,
            Vec<(u64, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = defenders
            .iter()
            .map(|(faction, _)| {
                (
                    faction.clone(),
                    root.neighbors
                        .get(&location)
                        .unwrap()
                        .iter()
                        .map(|n| {
                            (
                                Node::get_distance(*n, location, root),
                                Node::get_node_faction_reinforcements(
                                    *n,
                                    location,
                                    faction.clone(),
                                    root,
                                ),
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        EngagementPrep {
            turn: root.turn,
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
    turn: u64,
    attackers: HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    defenders: HashMap<Arc<Faction>, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>)>,
    aggressor: Arc<Faction>,
    objectives: HashMap<Arc<Faction>, Vec<ObjectiveFlavor>>,
    location: Key<Node>,
    duration: u64,
    victor: Vec<Arc<Faction>>,
    ship_status: HashMap<Key<ShipInstance>, (u64, Vec<u64>, ShipLocationFlavor)>,
    fleet_status: HashMap<Key<FleetInstance>, Key<Node>>,
}

impl Engagement {
    pub fn battle_cleanup(&self, root: &mut Root) {
        if self.victor.contains(&self.aggressor.clone()) {
            root.nodes.get_mut(self.location).unwrap().allegiance = self.aggressor.clone()
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
            if ship.hull > 0 {
                ship.repair_engagement()
            };
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
    fn get_value(self, node: (Key<Node>, &Node), faction: Arc<Faction>, root: &Root)
        -> Option<f32>;
}

//this method retrieves threat value generated by a given faction
impl Salience<polarity::Supply> for Arc<Faction> {
    const DEG_MULT: f32 = 0.5;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Arc<Faction>,
        root: &Root,
    ) -> Option<f32> {
        let node_strength: u64 = root.get_node_strength(nodeid, self.clone());
        //here we get the relations value -- the subject faction's opinion of the object faction, which will influence the threat value
        let relation = faction
            .relations
            .get(&FactionID::new_from_index(self.id))
            .unwrap();
        Some(node_strength)
            .filter(|&strength| strength != 0)
            .map(|strength| strength as f32 * relation)
    }
}

//NOTE: This is dummied out currently! We need to think about how threat demand works.
impl Salience<polarity::Demand> for Arc<Faction> {
    const DEG_MULT: f32 = 0.5;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Arc<Faction>,
        root: &Root,
    ) -> Option<f32> {
        None
    }
}

//this method tells us how much supply there is of a given resource in a given node
impl Salience<polarity::Supply> for Arc<Resource> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Arc<Faction>,
        root: &Root,
    ) -> Option<f32> {
        //NOTE: Currently this does not take input stockpiles of any kind into account. We may wish to change this.
        //we add up all the resource quantity in factory output stockpiles in the node
        let factorysupply: u64 = if node.allegiance == faction {
            node.factoryinstancelist
                .iter()
                .map(|factory| factory.get_resource_supply_total(root, self.clone()))
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
            .map(|(_, ship)| ship.get_resource_supply(root, self.clone()))
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
impl Salience<polarity::Demand> for Arc<Resource> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Arc<Faction>,
        root: &Root,
    ) -> Option<f32> {
        //add up resources from factory input stockpiles in node
        let factorydemand: u64 = if node.allegiance == faction {
            node.factoryinstancelist
                .iter()
                .map(|factory| factory.get_resource_demand_total(root, self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //add up resources from shipyard input stockpiles in node
        let shipyarddemand: u64 = if node.allegiance == faction {
            node.shipyardinstancelist
                .iter()
                .map(|shipyard| shipyard.get_resource_demand_total(root, self.clone()))
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
            .map(|(_, ship)| ship.get_resource_demand(root, self.clone()))
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
impl Salience<polarity::Supply> for Arc<ShipClass> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Arc<Faction>,
        root: &Root,
    ) -> Option<f32> {
        let sum = root
            .shipinstances
            .iter()
            .filter(|(_, s)| s.get_node(root) == nodeid)
            .filter(|(_, s)| s.allegiance == faction)
            .map(|(_, s)| s.get_shipclass_supply(root, self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

//this method tells us how much demand there is for a given shipclass in a given node
impl Salience<polarity::Demand> for Arc<ShipClass> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        (nodeid, node): (Key<Node>, &Node),
        faction: Arc<Faction>,
        root: &Root,
    ) -> Option<f32> {
        let sum = root
            .shipinstances
            .iter()
            .filter(|(_, s)| s.get_node(root) == nodeid)
            .filter(|(_, s)| s.allegiance == faction)
            .map(|(_, s)| s.get_shipclass_demand(root, self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

//TODO: make the logic apply more generally to stockpiles attached to ships

#[derive(Debug, Clone)]
pub struct GlobalSalience {
    pub factionsalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub resourcesalience: Vec<Vec<Vec<[f32; 2]>>>,
    pub shipclasssalience: Vec<Vec<Vec<[f32; 2]>>>,
}

#[derive(Debug)]
pub struct Root {
    pub config: Config,
    pub nodeflavors: Vec<Arc<NodeFlavor>>,
    pub nodes: Table<Node>,
    pub systems: Table<System>,
    pub edgeflavors: Table<EdgeFlavor>,
    pub edges: HashMap<(Key<Node>, Key<Node>), Key<EdgeFlavor>>,
    pub neighbors: HashMap<Key<Node>, Vec<Key<Node>>>,
    pub factions: Vec<Arc<Faction>>,
    pub wars: HashSet<(Arc<Faction>, Arc<Faction>)>,
    pub resources: Vec<Arc<Resource>>,
    pub hangarclasses: Vec<Arc<HangarClass>>,
    pub hangarinstances: Table<HangarInstance>,
    pub engineclasses: Table<EngineClass>,
    pub repairerclasses: Table<RepairerClass>,
    pub factoryclasses: Table<FactoryClass>,
    pub shipyardclasses: Table<ShipyardClass>,
    pub shipais: Table<ShipAI>,
    pub shipclasses: Vec<Arc<ShipClass>>,
    pub shipinstances: Table<ShipInstance>,
    pub shipinstancecounter: usize,
    pub fleetclasses: Table<FleetClass>,
    pub fleetinstances: Table<FleetInstance>,
    pub engagements: Table<Engagement>,
    pub globalsalience: GlobalSalience,
    pub turn: u64,
}

impl Root {
    /*pub fn balance_stockpiles(
        &mut self,
        nodeid: Key<Node>,
        faction: Arc<Faction>,
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

    pub fn balance_hangars(
        &mut self,
        nodeid: Key<Node>,
        faction: Arc<Faction>,
        salience_map: Vec<f32>,
    ) {
    }

    pub fn create_hangar(&mut self, hangarclass: Arc<HangarClass>) -> Key<HangarInstance> {
        //we call the hangarclass instantiate method, and feed it the parameters it wants
        self.hangarinstances.put(HangarClass::instantiate(
            hangarclass,
            self.hangarinstances.next_index,
            &self.shipclasses,
        ))
    }
    //this is the method for creating a ship
    //duh
    pub fn create_ship(
        &mut self,
        class: Arc<ShipClass>,
        location: ShipLocationFlavor,
        faction: Arc<Faction>,
    ) -> Key<ShipInstance> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        //let index_lock = RwLock::new(self.shipinstances);
        let new_ship = ShipClass::instantiate(
            class,
            location,
            faction,
            self.shipinstances.next_index,
            self,
        );
        dbg!(&new_ship);
        let ship_key = self.shipinstances.put(new_ship);
        assert_eq!(ship_key, self.shipinstances.get(ship_key).unwrap().id);
        ship_key
    }
    pub fn engagement_check(&self, nodeid: Key<Node>, actor: Arc<Faction>) -> Option<Arc<Faction>> {
        let factions = Node::get_node_factions(nodeid, self);
        if factions.iter().any(|f1| {
            factions.iter().any(|f2| {
                self.wars
                    .contains(&(f1.clone().min(f2.clone()), f2.clone().max(f1.clone())))
            })
        }) {
            Some(actor)
        } else {
            None
        }
    }
    pub fn internal_battle(&self, data: EngagementPrep) -> Engagement {
        //we determine how long the battle lasts
        //taking into account both absolute and relative armada sizes
        //scaled logarithmically according to the specified exponent
        //as well as the scaling factors applied by the objectives of parties involved
        //then we multiply by a random number from a normal distribution
        let attacker_rough_strength: i64 = (data
            .attackers
            .iter()
            .map(|(_, (_, ss))| {
                ss.iter()
                    .filter(|s| self.shipinstances.get(**s).unwrap().is_in_node(self))
                    .map(|s| {
                        self.shipinstances
                            .get(*s)
                            .unwrap()
                            .get_strength(self, self.config.battlescalars.avg_duration)
                    })
                    .sum::<u64>()
            })
            .sum::<u64>()
            + data
                .attackers
                .iter()
                .map(|(_, (fs, _))| {
                    fs.iter()
                        .map(|f| {
                            self.fleetinstances
                                .get(*f)
                                .unwrap()
                                .get_strength(self, self.config.battlescalars.avg_duration)
                        })
                        .sum::<u64>()
                })
                .sum::<u64>()) as i64;
        let defender_rough_strength: i64 = (data
            .defenders
            .iter()
            .map(|(_, (_, ss))| {
                ss.iter()
                    .filter(|s| self.shipinstances.get(**s).unwrap().is_in_node(self))
                    .map(|s| {
                        self.shipinstances
                            .get(*s)
                            .unwrap()
                            .get_strength(self, self.config.battlescalars.avg_duration)
                    })
                    .sum::<u64>()
            })
            .sum::<u64>()
            + data
                .defenders
                .iter()
                .map(|(_, (fs, _))| {
                    fs.iter()
                        .map(|f| {
                            self.fleetinstances
                                .get(*f)
                                .unwrap()
                                .get_strength(self, self.config.battlescalars.avg_duration)
                        })
                        .sum::<u64>()
                })
                .sum::<u64>()) as i64;
        let battle_size = (attacker_rough_strength + defender_rough_strength)
            - (attacker_rough_strength - defender_rough_strength).abs();

        let objective_duration_scalar: f32 = data
            .attackers
            .iter()
            .map(|(_, (fs, ss))| {
                let mut d = fs
                    .iter()
                    .map(|id| self.fleetinstances.get(*id).unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                        .map(|id| self.shipinstances.get(*id).unwrap().objectives.clone())
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
                        .map(|id| self.fleetinstances.get(*id).unwrap().objectives.clone())
                        .flatten()
                        .collect::<Vec<ObjectiveFlavor>>();
                    d.append(
                        &mut ss
                            .iter()
                            .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                            .map(|id| self.shipinstances.get(*id).unwrap().objectives.clone())
                            .flatten()
                            .collect::<Vec<ObjectiveFlavor>>(),
                    );
                    d
                })
                .flatten()
                .map(|of| of.get_scalars().durationscalar)
                .product::<f32>();

        let duration: u64 = (((battle_size as f32).log(self.config.battlescalars.duration_log_exp)
            + 300.0)
            * objective_duration_scalar
            * Normal::new(1.0, self.config.battlescalars.duration_dev)
                .unwrap()
                .sample(&mut rand::thread_rng()))
        .clamp(0.0, 2.0) as u64;

        //we take the reinforcement data from our engagementprep and convert the distances to the scaling factor for travel time
        //for what percentage of the battle's duration the unit will be present
        let scaled_attacker_reinforcements: HashMap<
            Arc<Faction>,
            Vec<(f32, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = data
            .attacker_reinforcements
            .iter()
            .map(|(faction, vec)| {
                (
                    faction.clone(),
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
            Arc<Faction>,
            Vec<(f32, (Vec<Key<FleetInstance>>, Vec<Key<ShipInstance>>))>,
        > = data
            .defender_reinforcements
            .iter()
            .map(|(faction, vec)| {
                (
                    faction.clone(),
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
                    .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                    .map(|shipid| {
                        self.shipinstances
                            .get(*shipid)
                            .unwrap()
                            .get_strength(self, duration)
                    })
                    .sum::<u64>()
                    + fs.iter()
                        .map(|f| {
                            self.fleetinstances
                                .get(*f)
                                .unwrap()
                                .get_strength(self, duration)
                        })
                        .sum::<u64>()
            })
            .sum::<u64>()
            + scaled_attacker_reinforcements
                .iter()
                .map(|(_, v)| {
                    v.iter()
                        .map(|(scalar, (fs, ss))| {
                            ((ss.iter()
                                .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                                .map(|s| {
                                    self.shipinstances
                                        .get(*s)
                                        .unwrap()
                                        .get_strength(self, duration)
                                })
                                .sum::<u64>()
                                + fs.iter()
                                    .map(|f| {
                                        self.fleetinstances
                                            .get(*f)
                                            .unwrap()
                                            .get_strength(self, duration)
                                    })
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
                    .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                    .map(|shipid| {
                        self.shipinstances
                            .get(*shipid)
                            .unwrap()
                            .get_strength(self, duration)
                    })
                    .sum::<u64>()
                    + fs.iter()
                        .map(|f| {
                            self.fleetinstances
                                .get(*f)
                                .unwrap()
                                .get_strength(self, duration)
                        })
                        .sum::<u64>()
            })
            .sum::<u64>()
            + scaled_defender_reinforcements
                .iter()
                .map(|(fid, v)| {
                    v.iter()
                        .map(|(scalar, (fs, ss))| {
                            ((ss.iter()
                                .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                                .map(|s| {
                                    self.shipinstances
                                        .get(*s)
                                        .unwrap()
                                        .get_strength(self, duration)
                                })
                                .sum::<u64>()
                                + fs.iter()
                                    .map(|f| {
                                        self.fleetinstances
                                            .get(*f)
                                            .unwrap()
                                            .get_strength(self, duration)
                                    })
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
                    .map(|id| self.fleetinstances.get(*id).unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                        .map(|id| self.shipinstances.get(*id).unwrap().objectives.clone())
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
                    .map(|id| self.fleetinstances.get(*id).unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|s| !self.shipinstances.get(**s).unwrap().is_in_fleet(self))
                        .map(|id| self.shipinstances.get(*id).unwrap().objectives.clone())
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
                .map(|(faction, _)| faction.battlescalar)
                .product::<f32>()
            * Normal::<f32>::new(1.0, self.config.battlescalars.attacker_chance_dev)
                .unwrap()
                .sample(&mut rand::thread_rng())
                .clamp(0.0, 2.0);

        let defender_chance: f32 = defender_strength as f32
            * defender_objective_difficulty
            * data
                .defenders
                .iter()
                .map(|(faction, _)| faction.battlescalar)
                .product::<f32>()
            * Normal::<f32>::new(1.0, self.config.battlescalars.defender_chance_dev)
                .unwrap()
                .sample(&mut rand::thread_rng())
                .clamp(0.0, 2.0);

        let (victor, victor_strength, victis_strength) = if attacker_chance > defender_chance {
            (
                vec![data.aggressor.clone()],
                attacker_strength as f32,
                defender_strength as f32,
            )
        } else {
            (
                data.defenders
                    .iter()
                    .map(|(faction, _)| faction.clone())
                    .collect(),
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

        let neighbors = self.neighbors.get(&data.location).unwrap();

        let fleet_status: HashMap<Key<FleetInstance>, Key<Node>> = all_fleets
            .iter()
            .map(|fleetid| {
                (
                    *fleetid,
                    self.fleetinstances
                        .get(*fleetid)
                        .unwrap()
                        .navigate(self, neighbors)
                        .unwrap_or(data.location),
                )
            })
            .collect();

        let duration_damage_rand = Normal::<f32>::new(1.0, self.config.battlescalars.damage_dev)
            .unwrap()
            .sample(&mut rand::thread_rng())
            .clamp(0.0, 1.0);

        //NOTE: Maybe have the lethality scaling over battle duration be logarithmic? Maybe modder-specified?
        let ship_status: HashMap<Key<ShipInstance>, (u64, Vec<u64>, ShipLocationFlavor)> = {
            all_ships
                .iter()
                .map(|shipid| {
                    let ship = self.shipinstances.get(*shipid).unwrap();
                    let rand_factor =
                        Normal::<f32>::new(0.25, self.config.battlescalars.damage_dev)
                            .unwrap()
                            .sample(&mut rand::thread_rng())
                            .clamp(0.0, 10.0);
                    if !victor.contains(&ship.allegiance) {
                        let new_location = if ship.is_in_node(self) {
                            ShipLocationFlavor::Node(
                                ship.navigate(self, neighbors).unwrap_or(data.location),
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
                        let damage = (((ship.class.basehull as f32
                            * (victor_strength / victis_strength)
                            * (duration as f32
                                * self.config.battlescalars.duration_damage_scalar
                                * duration_damage_rand)
                            * rand_factor
                            * self.config.battlescalars.vae_victis)
                            + (self.config.battlescalars.base_damage
                                * (victor_strength / victis_strength)
                                * (duration as f32
                                    * self.config.battlescalars.duration_damage_scalar
                                    * duration_damage_rand)
                                * rand_factor
                                * self.config.battlescalars.vae_victis))
                            / (ship.class.toughnessscalar + ship.class.battleescapescalar))
                            as u64;
                        let engine_damage: Vec<u64> = ship
                            .engines
                            .iter()
                            .filter(|e| e.health.is_some())
                            .map(|e| {
                                ((damage as f32
                                    * Normal::<f32>::new(
                                        1.0,
                                        self.config.battlescalars.damage_dev,
                                    )
                                    .unwrap()
                                    .sample(&mut rand::thread_rng())
                                    .clamp(0.0, 2.0)
                                    * self.config.battlescalars.engine_damage_scalar)
                                    / e.toughnessscalar) as u64
                            })
                            .collect();
                        (*shipid, (damage, engine_damage, new_location))
                    } else {
                        let new_location = if all_fleets.contains(
                            &ship
                                .get_fleet(self)
                                .unwrap_or(Key::new_from_index(usize::MAX)),
                        ) {
                            ship.location
                        } else {
                            ShipLocationFlavor::Node(data.location)
                        };
                        //we do basically the same thing for winning ships
                        //except that the strength ratio is reversed
                        //we use the damage multiplier for winners instead of losers
                        //and we don't take battleescapescalar into account
                        let damage = (((ship.class.basehull as f32
                            * (victis_strength / victor_strength)
                            * (duration as f32
                                * self.config.battlescalars.duration_damage_scalar
                                * duration_damage_rand)
                            * rand_factor
                            * self.config.battlescalars.vae_victor)
                            + (self.config.battlescalars.base_damage
                                * (victis_strength / victor_strength)
                                * (duration as f32
                                    * self.config.battlescalars.duration_damage_scalar
                                    * duration_damage_rand)
                                * rand_factor
                                * self.config.battlescalars.vae_victor))
                            / ship.class.toughnessscalar)
                            as u64;
                        let engine_damage: Vec<u64> = ship
                            .engines
                            .iter()
                            .filter(|e| e.health.is_some())
                            .map(|e| {
                                ((damage as f32
                                    * Normal::<f32>::new(
                                        1.0,
                                        self.config.battlescalars.damage_dev,
                                    )
                                    .unwrap()
                                    .sample(&mut rand::thread_rng())
                                    .clamp(0.0, 2.0)
                                    * self.config.battlescalars.engine_damage_scalar)
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
                self.nodes.get(data.location).unwrap().visiblename
            ),
            turn: data.turn,
            attackers: data.attackers,
            defenders: data.defenders,
            aggressor: data.aggressor.clone(),
            objectives: HashMap::new(),
            location: data.location,
            duration: duration,
            victor: victor,
            ship_status: ship_status,
            fleet_status: fleet_status,
        }
    }
    pub fn remove_dead(&mut self) {
        let mut dead: Vec<Key<ShipInstance>> = self
            .shipinstances
            .iter()
            .filter(|(_, s)| s.hull == 0)
            .map(|(k, v)| *k)
            .collect();
        dead.iter_mut().for_each(|id| {
            let ship = self.shipinstances.get(*id).unwrap().clone();
            ship.kill(&mut self.shipinstances, &self.hangarinstances);
        });
        self.shipinstances.retain(|_, s| s.hull > 0);
    }
    pub fn disband_fleets(&mut self) {
        let dead = self
            .fleetinstances
            .iter()
            .filter(|(_, fleet)| {
                let class = self.fleetclasses.get(fleet.fleetclass).unwrap();
                ((fleet.get_strength(self, self.config.battlescalars.avg_duration) as f32)
                    < (fleet.idealstrength as f32 * class.disbandthreshold))
                    && !fleet.phantom
            })
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for id in dead {
            let fleet = self.fleetinstances.get(id).unwrap().clone();
            fleet.disband(self);
        }
        let remaining: Vec<Key<FleetInstance>> = self
            .fleetinstances
            .iter()
            .filter(|(_, fleet)| fleet.phantom || !fleet.get_daughters(self).is_empty())
            .map(|(k, _)| *k)
            .collect();
        self.fleetinstances.retain(|id, _| remaining.contains(id));
    }
    //we get the military strength of a node for a given faction by filtering down the global ship list by node and faction allegiance, then summing their strength values
    fn get_node_strength(&self, nodeid: Key<Node>, faction: Arc<Faction>) -> u64 {
        self.shipinstances
            .iter()
            .filter(|(_, ship)| ship.get_node(&self) == nodeid)
            .filter(|(_, ship)| ship.allegiance == faction)
            .map(|(_, ship)| ship.get_strength(&self, self.config.battlescalars.avg_duration))
            .sum()
    }
    //oh god
    //NOTE: I removed the "+ Copy" from this so that it can use Arc<ShipClass>es or the like. Not sure if that's a problem.
    pub fn calculate_values<S: Salience<P> + Clone, P: Polarity>(
        //we need a salience, which is the type of resource or shipclass or whatever we're calculating values for
        //and the faction for which we're calculating values
        //and we specify the number of times we want to calculate these values, (NOTE: uncertain) i.e. the number of edges we'll propagate across
        &self,
        salience: S,
        subject_faction: Arc<Faction>,
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
                    .clone()
                    .get_value((id, node), subject_faction.clone(), &self)
                    .map(|v| (id, v))
            })
            .collect();
        //this map contains the amount of threat that exists from each faction, in each node, from the perspective of the subject faction
        //Length equals all nodes
        //This is a subjective map for subject faction
        let tagged_threats: Vec<HashMap<Arc<Faction>, f32>> = self
            .nodes
            .iter()
            .map(|(_, node)| {
                //we iterate over the node's threat listing, and get the threat for each faction as perceived by the subject faction -- that is, multiplied by the subject faction's relations with that faction
                node.threat
                    .iter()
                    .map(|(faction, t)| {
                        let value = t * subject_faction
                            .relations
                            .get(&FactionID::new_from_index(faction.id))
                            .unwrap();
                        (faction.clone(), value)
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
    //NOTE: This method will shortly be rendered obsolete by calculate_global_faction_salience
    pub fn update_node_threats(&mut self, n_steps: usize) {
        let faction_threat: Vec<(Arc<Faction>, Vec<f32>)> = self
            .factions
            .iter()
            .map(|faction| {
                let v = self.calculate_values::<Arc<Faction>, polarity::Supply>(
                    faction.clone(),
                    faction.clone(),
                    n_steps,
                );
                (faction.clone(), v)
            })
            .collect();
        faction_threat.iter().for_each(|(faction, threat_list)| {
            threat_list
                .iter()
                .zip(self.nodes.iter_mut())
                .for_each(|(&threat_v, (_, node))| {
                    node.threat.insert(faction.clone(), threat_v).unwrap();
                })
        })
    }
    //NOTE: I don't know why n_iters is a usize, but that's what calculate_values wants for some reason
    pub fn calculate_global_faction_salience(&self, n_iters: usize) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .iter()
            .map(|subjectfaction| {
                self.factions
                    .iter()
                    .map(|objectfaction| {
                        let supply = self.calculate_values::<Arc<Faction>, polarity::Supply>(
                            objectfaction.clone(),
                            subjectfaction.clone(),
                            n_iters,
                        );
                        let demand = self.calculate_values::<Arc<Faction>, polarity::Demand>(
                            objectfaction.clone(),
                            subjectfaction.clone(),
                            n_iters,
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
    pub fn calculate_global_resource_salience(&self, n_iters: usize) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .iter()
            .map(|faction| {
                self.resources
                    .iter()
                    .map(|resource| {
                        let supply = self.calculate_values::<Arc<Resource>, polarity::Supply>(
                            resource.clone(),
                            faction.clone(),
                            n_iters,
                        );
                        let demand = self.calculate_values::<Arc<Resource>, polarity::Demand>(
                            resource.clone(),
                            faction.clone(),
                            n_iters,
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
    pub fn calculate_global_shipclass_salience(&self, n_iters: usize) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .iter()
            .map(|faction| {
                self.shipclasses
                    .iter()
                    .map(|shipclass| {
                        let supply = self.calculate_values::<Arc<ShipClass>, polarity::Supply>(
                            shipclass.clone(),
                            faction.clone(),
                            n_iters,
                        );
                        let demand = self.calculate_values::<Arc<ShipClass>, polarity::Demand>(
                            shipclass.clone(),
                            faction.clone(),
                            n_iters,
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
        //increment turn counter
        self.turn += 1;
        println!("It is now turn {}.", self.turn);

        //reset all ships' engines
        self.shipinstances
            .iter_mut()
            .for_each(|(_, s)| s.reset_movement());

        //run all ship repairers
        self.shipinstances
            .iter_mut()
            .for_each(|(_, s)| s.repair_turn());

        //process all factories
        self.nodes
            .iter_mut()
            .for_each(|(_, n)| n.process_factories());
        self.shipinstances
            .iter_mut()
            .for_each(|(_, s)| s.process_factories());

        //process all shipyards
        self.nodes
            .iter_mut()
            .for_each(|(_, n)| n.process_shipyards());
        self.shipinstances
            .iter_mut()
            .for_each(|(_, s)| s.process_shipyards());

        //plan ship creation
        let ship_plan_list: Vec<(Arc<ShipClass>, ShipLocationFlavor, Arc<Faction>)> = self
            .nodes
            .iter_mut()
            .map(|(nid, n)| n.plan_ships(*nid, &self.shipclasses))
            .chain(
                self.shipinstances
                    .iter_mut()
                    .map(|(_, s)| s.plan_ships(&self.shipclasses)),
            )
            .flatten()
            .collect();

        //create queued ships
        let n_newships = ship_plan_list
            .iter()
            .map(|(id, location, faction)| self.create_ship(id.clone(), *location, faction.clone()))
            .count();
        println!("Built {} new ships.", n_newships);

        //propagate threat values

        //propagate saliences, create salience map

        //run operation management logic

        //move ships, one edge at a time
        //running battle checks and stockpile balancing with each traversal
        /*
        self.shipinstances.iter_mut().for_each(|(_, shipinstance)| {
            if let Some(destination) = shipinstance.maneuver(self) {
                if let Some(aggressor) = self.engagement_check(destination, shipinstance.allegiance)
                {
                    let engagement = self.internal_battle(EngagementPrep::engagement_prep(
                        self,
                        destination,
                        shipinstance.allegiance,
                    ));
                    self.engagements.put(engagement);
                }
            }
        });
        */

        //move fleets, one edge at a time
        //running battle checks and stockpile balancing with each traversal

        //run defection logic

        //run diplomacy logic

        //transmit root data to frontend

        for _ in 0..10 {
            self.update_node_threats(10);
        }

        //NOTE: I don't remember what this is for
        //I don't *think* it's doing anything that actually matters, but Chesterton's Fence?
        self.nodes.iter().for_each(|(_, node)| {
            let mut threat_list: Vec<(Arc<Faction>, f32)> = node
                .threat
                .iter()
                .map(|(faction, v)| (faction.clone(), *v))
                .collect();
            threat_list.sort_by_key(|(faction, _)| faction.clone());
        });
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
