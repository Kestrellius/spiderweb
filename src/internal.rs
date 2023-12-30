use ordered_float::NotNan;
use rand::prelude::*;
use rand_distr::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::iter;
use std::mem;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::{RwLock, RwLockWriteGuard};
use std::time::Instant;

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
    pub faction_prop_iters: usize, //number of edges across which this salience will propagate during a turn
    pub resource_prop_iters: usize,
    pub shipclass_prop_iters: usize,
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

#[derive(Debug, Clone)]
pub struct NodeFlavor {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
}

impl PartialEq for NodeFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for NodeFlavor {}

impl Ord for NodeFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for NodeFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for NodeFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct NodeMut {
    pub visibility: bool,
    pub flavor: Arc<NodeFlavor>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub units: Vec<Unit>,
    pub factoryinstancelist: Vec<FactoryInstance>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub allegiance: Arc<Faction>, //faction that currently holds the node
    pub efficiency: f32, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub threat: HashMap<Arc<Faction>, f32>,
    pub already_balanced: bool,
}

#[derive(Debug)]
pub struct Node {
    pub id: usize,
    pub visiblename: String, //location name as shown to player
    pub position: [i64; 3], //node's position in 3d space; this is used for autogenerating skyboxes and determining reinforcement delay between nodes
    pub description: String,
    pub environment: String, //name of the FRED environment to use for missions set in this node
    pub bitmap: Option<(String, f32)>, //name of the bitmap used to depict this node in other nodes' FRED environments, and a size factor
    pub mutables: RwLock<NodeMut>,
}

trait Locality {
    fn get_distance(&self, rhs: Arc<Node>) -> u64;
    fn plan_ships(
        &self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)>;
}

impl Locality for Arc<Node> {
    fn get_distance(&self, rhs: Arc<Node>) -> u64 {
        let self_pos = self.position;
        let rhs_pos = rhs.position;
        (((self_pos[0] - rhs_pos[0]) + (self_pos[1] - rhs_pos[1]) + (self_pos[2] - rhs_pos[2]))
            as f32)
            .sqrt() as u64
    }
    fn plan_ships(
        &self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)> {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        let allegiance = mutables.allegiance.clone();
        mutables
            .shipyardinstancelist
            .iter_mut()
            .map(|shipyard| {
                let ship_plans = shipyard.plan_ships(efficiency, shipclasses);
                //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                ship_plans
                    .iter()
                    .map(|ship_plan| {
                        (
                            ship_plan.clone(),
                            UnitLocation::Node(self.clone()),
                            allegiance.clone(),
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

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Node {
    fn get_strength(&self, faction: Arc<Faction>, time: u64) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_strength(time))
            .sum()
    }
    pub fn get_node_forces(
        &self,
        root: &Root,
    ) -> HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)> {
        root.factions
            .iter()
            .map(|faction| {
                let ships: Vec<Arc<ShipInstance>> = self
                    .mutables
                    .read()
                    .unwrap()
                    .units
                    .iter()
                    .filter(|unit| unit.get_allegiance() == *faction)
                    .filter_map(|unit| unit.get_ship())
                    .collect();
                let fleets: Vec<Arc<FleetInstance>> = self
                    .mutables
                    .read()
                    .unwrap()
                    .units
                    .iter()
                    .filter(|unit| unit.get_allegiance() == *faction)
                    .filter_map(|unit| unit.get_fleet())
                    .collect();
                (faction.clone(), (fleets, ships))
            })
            .filter(|(_, (_, ships))| ships.len() > 0)
            .collect()
    }
    pub fn get_node_factions(&self, root: &Root) -> Vec<Arc<Faction>> {
        root.factions
            .iter()
            .filter(|faction| {
                !self
                    .mutables
                    .read()
                    .unwrap()
                    .units
                    .iter()
                    .filter(|unit| unit.get_allegiance() == **faction)
                    .collect::<Vec<_>>()
                    .is_empty()
            })
            .cloned()
            .collect()
    }
    pub fn get_node_faction_reinforcements(
        &self,
        destination: Arc<Node>,
        factionid: Arc<Faction>,
        root: &Root,
    ) -> (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>) {
        let top_level_units: Vec<Unit> = self
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.get_allegiance() == factionid)
            .filter(|unit| {
                unit.destinations_check(root, &vec![destination.clone()])
                    .is_some()
            })
            .cloned()
            .collect();
        let daughter_units: Vec<Unit> = top_level_units
            .iter()
            .map(|unit| unit.get_traversal_checked_daughters(root, destination.clone()))
            .flatten()
            .collect();
        let (ship_units, fleet_units): (Vec<Unit>, Vec<Unit>) = top_level_units
            .into_iter()
            .chain(daughter_units.into_iter())
            .partition(|unit| unit.is_ship());
        (
            fleet_units
                .iter()
                .map(|unit| unit.get_fleet().unwrap())
                .collect(),
            ship_units
                .iter()
                .map(|unit| unit.get_ship().unwrap())
                .collect(),
        )
    }
    pub fn get_node_faction_forces(
        &self,
        faction: Arc<Faction>,
    ) -> (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>) {
        let ships: Vec<Arc<ShipInstance>> = self
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.mutables.read().unwrap().allegiance == faction)
            .collect();
        let fleets: Vec<Arc<FleetInstance>> = self
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter_map(|unit| unit.get_fleet())
            .filter(|fleet| fleet.mutables.read().unwrap().allegiance == faction)
            .collect();
        (fleets, ships)
    }
    pub fn get_shipclass_num(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.class.id == shipclass.id)
            .count() as u64
    }
    pub fn get_system(&self, root: &Root) -> Option<Arc<System>> {
        let system = root.systems.iter().find(|system| {
            system
                .nodes
                .iter()
                .find(|sys_node| sys_node.id == self.id)
                .is_some()
        });
        match system {
            Some(sys) => Some(sys.clone()),
            None => None,
        }
    }
    pub fn is_in_system(&self, system: Arc<System>) -> bool {
        system
            .nodes
            .iter()
            .find(|sys_node| sys_node.id == self.id)
            .is_some()
    }
    pub fn process_factories(&self) {
        let efficiency = self.mutables.read().unwrap().efficiency;
        self.mutables
            .write()
            .unwrap()
            .factoryinstancelist
            .iter_mut()
            .for_each(|f| f.process(efficiency));
    }
    pub fn process_shipyards(&self) {
        let efficiency = self.mutables.read().unwrap().efficiency;
        self.mutables
            .write()
            .unwrap()
            .shipyardinstancelist
            .iter_mut()
            .for_each(|sy| sy.process(efficiency));
    }
}

#[derive(Debug, Clone)]
pub struct System {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub nodes: Vec<Arc<Node>>,
}

impl PartialEq for System {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for System {}

impl Ord for System {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for System {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for System {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct EdgeFlavor {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
}

impl PartialEq for EdgeFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for EdgeFlavor {}

impl Ord for EdgeFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for EdgeFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for EdgeFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edges {
    hyperlinks: HashSet<(Arc<Node>, Arc<Node>, Arc<EdgeFlavor>)>, //list of links between nodes
    neighbormap: HashMap<Arc<Node>, Vec<Arc<Node>>>, //NOTE: investigate. Map of which nodes belong to which systems, for purposes of generating all-to-all links
}

impl Edges {
    //this creates an edge between two nodes, and adds each node to the other's neighbor map
    fn insert(&mut self, a: Arc<Node>, b: Arc<Node>, flavor: Arc<EdgeFlavor>) {
        assert_ne!(a.clone(), b.clone());
        self.hyperlinks
            .insert((a.clone().max(b.clone()), a.clone().min(b.clone()), flavor));
        self.neighbormap
            .entry(a.clone())
            .or_insert_with(|| Vec::new())
            .push(b.clone());
        self.neighbormap
            .entry(b)
            .or_insert_with(|| Vec::new())
            .push(a);
    }
    /*fn insert_with_distance(&mut self, root: &mut Root, a: Arc<Node>, b: Arc<Node>, distance: u64) {
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
    pub unit_vol: u64, //how much volume (in e.g. cubic meters) one unit of this resource takes up; this is intended only for player-facing use
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

pub trait Stockpileness {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64>;
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64;
    fn get_capacity(&self) -> u64;
    fn get_fullness(&self) -> u64;
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>>;
    fn get_resource_supply(&self, resourceid: Arc<Resource>) -> u64;
    fn get_resource_demand(&self, resourceid: Arc<Resource>) -> u64;
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64;
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64;
    fn transfer<S: Stockpileness>(&mut self, rhs: &mut S, cargo: Arc<Resource>, quantity: u64) {
        let extracted = self.remove(cargo.clone(), quantity);
        let remainder = rhs.insert(cargo.clone(), extracted);
        self.insert(cargo, remainder);
    }
}

//this is a horrible incomprehensible nightmare that Amaryllis put me through for some reason
//okay, so, a year later, what this actually does is that it takes two individual stockpiles and allows them to function together as a single stockpile
impl<A: Stockpileness, B: Stockpileness> Stockpileness for (A, B) {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.0
            .collate_contents()
            .iter()
            .chain(self.1.collate_contents().iter())
            .map(|(resource, value)| (resource.clone(), *value))
            .collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        self.0.get_resource_num(cargo.clone()) + self.1.get_resource_num(cargo.clone())
    }
    fn get_capacity(&self) -> u64 {
        self.0.get_capacity() + self.1.get_capacity()
    }
    fn get_fullness(&self) -> u64 {
        self.0.get_fullness() + self.1.get_fullness()
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        //self.0
        //    .get_allowed()
        //    .iter()
        //    .chain(self.1.get_allowed().iter())
        //    .collect()
        Some(Vec::new())
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        self.0.get_resource_supply(resource.clone()) + self.1.get_resource_supply(resource.clone())
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        self.0.get_resource_demand(resource.clone()) + self.1.get_resource_demand(resource.clone())
    }
    fn insert(&mut self, _cargo: Arc<Resource>, quantity: u64) -> u64 {
        quantity
    }
    fn remove(&mut self, _cargo: Arc<Resource>, _quantity: u64) -> u64 {
        0
    }
}

//a unipotent resource stockpile can contain only one type of resource, and it cannot contain ship instances
//however, the quantity of resource specified in the rate field may be added to or removed from the stockpile under various circumstances,
//such as once every turn, depending on how it's used
#[derive(Debug, Clone, PartialEq)]
pub struct UnipotentStockpile {
    pub visibility: bool,
    pub resourcetype: Arc<Resource>,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for UnipotentStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((self.resourcetype.clone(), self.contents)).collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resourcetype {
            self.contents
        } else {
            0
        }
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self) -> u64 {
        self.contents
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        Some(vec![self.resourcetype.clone()])
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resourcetype {
            (self.contents).saturating_sub(self.target)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resourcetype {
            self.target.saturating_sub(self.contents)
        } else {
            0
        }
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resourcetype {
            let old_contents = self.contents;
            let remainder = quantity.saturating_sub(self.capacity - old_contents);
            self.contents += quantity - remainder;
            assert!(self.contents <= self.capacity);
            assert_eq!(self.contents + remainder, old_contents + quantity);
            remainder
        } else {
            quantity
        }
    }
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resourcetype {
            let _old_contents = self.contents;
            let remainder = quantity.saturating_sub(self.contents);
            self.contents -= quantity - remainder;
            quantity - remainder
        } else {
            0
        }
    }
}

impl UnipotentStockpile {
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
    pub contents: HashMap<Arc<Resource>, u64>,
    pub allowed: Option<Vec<Arc<Resource>>>,
    pub target: u64,
    pub capacity: u64,
    pub propagate: bool,
}

impl Stockpileness for PluripotentStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.contents.clone()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        *self.contents.get(&cargo).unwrap_or(&0)
    }
    fn get_capacity(&self) -> u64 {
        self.capacity.clone()
    }
    fn get_fullness(&self) -> u64 {
        self.contents.iter().map(|(_, value)| value).sum::<u64>()
    }
    //NOTE: Partially dummied out currently; waiting on removal of ship-carrying ability from stockpiles
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        match &self.allowed {
            Some(resource_allowed) => Some(resource_allowed.clone()),
            None => None,
        }
    }
    //unlike other places, here in pluripotent stockpiles we don't take target into account when calculating supply
    //thus, items in pluripotent stockpiles always emit supply, even if the stockpile still wants more
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        self.get_resource_num(resource.clone())
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        if self
            .get_allowed()
            .unwrap_or(vec![resource.clone()])
            .contains(&resource.clone())
        {
            self.target.saturating_sub(self.get_fullness())
        } else {
            0
        }
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if self
            .allowed
            .clone()
            .map(|allowed| allowed.contains(&cargo))
            .unwrap_or(true)
        {
            let old_fullness = self.get_fullness();
            let remainder = quantity.saturating_sub(self.capacity - old_fullness);
            *self.contents.get_mut(&cargo).unwrap() += quantity - remainder;
            assert!(self.get_fullness() <= self.capacity);
            assert_eq!(self.get_fullness() + remainder, old_fullness + quantity);
            remainder
        } else {
            quantity
        }
    }
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if self
            .allowed
            .clone()
            .map(|allowed| allowed.contains(&cargo))
            .unwrap_or(true)
        {
            let old_contents = self.contents.get(&cargo).unwrap();
            let remainder = quantity.saturating_sub(*old_contents);
            *self.contents.get_mut(&cargo).unwrap() -= quantity - remainder;
            quantity - remainder
        } else {
            0
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
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((
            self.resourcetype.clone(),
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resourcetype {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self) -> u64 {
        self.contents.load(atomic::Ordering::SeqCst)
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        Some(vec![self.resourcetype.clone()])
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resourcetype {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, _resourceid: Arc<Resource>) -> u64 {
        0
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resourcetype {
            let old_contents = self.contents.load(atomic::Ordering::SeqCst);
            let remainder = quantity.saturating_sub(self.capacity - old_contents);
            self.contents
                .fetch_add(quantity - remainder, atomic::Ordering::SeqCst);
            assert!(self.contents.load(atomic::Ordering::SeqCst) <= self.capacity);
            assert_eq!(
                self.contents.load(atomic::Ordering::SeqCst) + remainder,
                old_contents + quantity
            );
            remainder
        } else {
            quantity
        }
    }
    fn remove(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resourcetype {
            let old_contents = self.contents.load(atomic::Ordering::SeqCst);
            let remainder = quantity.saturating_sub(old_contents);
            self.contents
                .fetch_sub(quantity - remainder, atomic::Ordering::SeqCst);
            quantity - remainder
        } else {
            0
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
    pub target: u64, //volume the hangar wants to hold; this is usually equal to capacity
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
                root.shipclasses[shipclassid.index].get_ideal_strength(root) * v
            })
            .sum()
    }
    pub fn instantiate(
        class: Arc<Self>,
        mother: Arc<ShipInstance>,
        _shipclasses: &Vec<Arc<ShipClass>>,
        counter: &Arc<AtomicU64>,
    ) -> HangarInstance {
        let index = counter.fetch_add(1, atomic::Ordering::Relaxed);
        HangarInstance {
            id: index,
            class: class.clone(),
            mother: mother.clone(),
            mutables: RwLock::new(HangarInstanceMut {
                visibility: class.visibility,
                contents: Vec::new(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HangarInstanceMut {
    pub visibility: bool,
    pub contents: Vec<Unit>,
}

#[derive(Debug)]
pub struct HangarInstance {
    pub id: u64,
    pub class: Arc<HangarClass>,
    pub mother: Arc<ShipInstance>,
    pub mutables: RwLock<HangarInstanceMut>,
}

impl PartialEq for HangarInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for HangarInstance {}

impl Ord for HangarInstance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for HangarInstance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for HangarInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl HangarInstance {
    pub fn get_strength(&self, time: u64) -> u64 {
        let contents_strength = self
            .mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .map(|ship| ship.get_strength(time))
            .sum::<u64>() as f32;
        let contents_vol = self
            .mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .map(|unit| unit.get_volume())
            .sum::<u64>() as f32;
        //we calculate how much of its complement the hangar can launch during a battle a certain number of seconds long
        let launch_mod = ((contents_vol / self.class.launch_volume as f32)
            * (time as f32 / self.class.launch_interval as f32))
            .clamp(0.0, 1.0);
        (contents_strength * launch_mod) as u64
    }
    pub fn get_fullness(&self) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .map(|unit| unit.get_volume())
            .sum()
    }
    pub fn get_shipclass_num(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.class == shipclass)
            .collect::<Vec<_>>()
            .len()
            .try_into()
            .unwrap()
    }
    pub fn get_shipclass_supply(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.class == shipclass)
            .map(|_| shipclass.hangarvol)
            .sum()
    }
    pub fn get_shipclass_demand(&self, shipclass: Arc<ShipClass>) -> u64 {
        let ideal_num = self
            .class
            .ideal
            .get(&ShipClassID::new_from_index(shipclass.id))
            .unwrap_or(&0);
        (ideal_num.saturating_sub(self.get_shipclass_num(shipclass.clone()))) * shipclass.hangarvol
    }
}

pub trait ResourceProcess {
    fn get_state(&self) -> FactoryState;
    fn get_resource_supply_total(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64;
    fn get_target_total(&self) -> u64;
    fn get_output_resource_num(&self, cargo: Arc<Resource>) -> u64;
}

#[derive(Debug, Clone)]
pub struct EngineClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub basehealth: Option<u64>,
    pub toughnessscalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl PartialEq for EngineClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for EngineClass {}

impl Ord for EngineClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for EngineClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for EngineClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl EngineClass {
    pub fn instantiate(class: Arc<Self>) -> EngineInstance {
        EngineInstance {
            engineclass: class.clone(),
            visibility: class.visibility,
            basehealth: class.basehealth,
            health: class.basehealth,
            toughnessscalar: class.toughnessscalar,
            inputs: class.inputs.clone(),
            forbidden_nodeflavors: class.forbidden_nodeflavors.clone(),
            forbidden_edgeflavors: class.forbidden_edgeflavors.clone(),
            speed: class.speed,
            cooldown: class.cooldown,
            last_move_turn: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineInstance {
    engineclass: Arc<EngineClass>,
    visibility: bool,
    basehealth: Option<u64>,
    health: Option<u64>,
    toughnessscalar: f32,
    inputs: Vec<UnipotentStockpile>,
    forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
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
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn get_output_resource_num(&self, _cargo: Arc<Resource>) -> u64 {
        0
    }
}

impl EngineInstance {
    fn check_engine(
        &self,
        root: &Root,
        location: Arc<Node>,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<(Vec<Arc<Node>>, u64)> {
        if (self.health != Some(0))
            && (root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn > self.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            let viable = destinations
                .iter()
                .filter(|destination| {
                    self.nav_check(root, location.clone(), destination.clone().clone())
                })
                .cloned()
                .collect();
            Some((viable, self.speed))
        } else {
            None
        }
    }
    //this is run once per turn for a given engine; it checks to see if the engine has enough resources to run this turn and whether it's off cooldown
    //then consumes stockpile resources, and returns the engine's speed
    //we'll need to reset movement_left to max at the start of the turn
    fn process_engine(
        &mut self,
        root: &Root,
        location: Arc<Node>,
        destination: Arc<Node>,
    ) -> Option<u64> {
        if (self.health != Some(0))
            && (root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn > self.cooldown)
            && (self.get_state() == FactoryState::Active)
            && (self.nav_check(root, location, destination))
        {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.last_move_turn = root.turn.load(atomic::Ordering::Relaxed);
            Some(self.speed)
        } else {
            None
        }
    }
    fn nav_check(&self, root: &Root, location: Arc<Node>, destination: Arc<Node>) -> bool {
        !self
            .forbidden_nodeflavors
            .contains(&destination.mutables.read().unwrap().flavor)
            && root
                .edges
                .get(&(
                    location.clone().min(destination.clone()),
                    destination.max(location),
                ))
                .map(|edge| !self.forbidden_edgeflavors.contains(edge))
                .unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
pub struct RepairerClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl PartialEq for RepairerClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RepairerClass {}

impl Ord for RepairerClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for RepairerClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for RepairerClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl RepairerClass {
    pub fn instantiate(class: Arc<Self>) -> RepairerInstance {
        RepairerInstance {
            repairerclass: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
            repair_points: class.repair_points,
            repair_factor: class.repair_factor,
            engine_repair_points: class.engine_repair_points,
            engine_repair_factor: class.engine_repair_factor,
            per_engagement: class.per_engagement,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepairerInstance {
    repairerclass: Arc<RepairerClass>,
    visibility: bool,
    inputs: Vec<UnipotentStockpile>,
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
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn get_output_resource_num(&self, _cargo: Arc<Resource>) -> u64 {
        0
    }
}

impl RepairerInstance {
    fn process(&mut self) {
        self.inputs
            .iter_mut()
            .for_each(|input| input.input_process());
    }
}

#[derive(Debug, Clone)]
pub struct FactoryClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl PartialEq for FactoryClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FactoryClass {}

impl Ord for FactoryClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for FactoryClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for FactoryClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl FactoryClass {
    pub fn instantiate(class: Arc<Self>) -> FactoryInstance {
        FactoryInstance {
            factoryclass: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
            outputs: class.outputs.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactoryInstance {
    //this is an actual factory, derived from a factory class
    factoryclass: Arc<FactoryClass>,
    visibility: bool,
    inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
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
    fn get_resource_supply_total(&self, resource: Arc<Resource>) -> u64 {
        self.outputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>()
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
            + self
                .outputs
                .iter()
                .filter(|sp| sp.propagate)
                .map(|sp| sp.get_resource_demand(resource.clone()))
                .sum::<u64>()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum::<u64>()
            + self.outputs.iter().map(|sp| sp.target).sum::<u64>()
    }
    fn get_output_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        self.outputs
            .iter()
            .map(|sp| sp.get_resource_num(cargo.clone()))
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

#[derive(Debug, Clone)]
pub struct ShipyardClass {
    pub id: usize,
    pub visiblename: Option<String>,
    pub description: Option<String>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<ShipClassID, u64>,
    pub constructrate: u64,
    pub efficiency: f32,
}

impl PartialEq for ShipyardClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipyardClass {}

impl Ord for ShipyardClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipyardClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipyardClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl ShipyardClass {
    pub fn instantiate(class: Arc<Self>, shipclasses: &Vec<Arc<ShipClass>>) -> ShipyardInstance {
        ShipyardInstance {
            shipyardclass: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
            outputs: class
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
            constructrate: class.constructrate,
            efficiency: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShipyardInstance {
    shipyardclass: Arc<ShipyardClass>,
    visibility: bool,
    inputs: Vec<UnipotentStockpile>,
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
    fn get_resource_supply_total(&self, _resource: Arc<Resource>) -> u64 {
        0
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
    fn get_output_resource_num(&self, _cargo: Arc<Resource>) -> u64 {
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

    fn try_choose_ship(&mut self, _shipclasses: &Vec<Arc<ShipClass>>) -> Option<Arc<ShipClass>> {
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
    pub id: usize,
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<ShipClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
}

impl PartialEq for ShipAI {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipAI {}

impl Ord for ShipAI {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipAI {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipAI {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum UnitLocation {
    Node(Arc<Node>),
    Fleet(Arc<FleetInstance>),
    Hangar(Arc<HangarInstance>),
}

impl UnitLocation {
    fn check_insert(&self, unit: Unit) -> bool {
        match self {
            UnitLocation::Node(_node) => true,
            UnitLocation::Fleet(_fleet) => true,
            UnitLocation::Hangar(hangar) => {
                unit.get_volume() <= hangar.class.capacity - hangar.get_fullness()
            }
        }
    }
    fn check_remove(&self, unit: Unit) -> bool {
        match self {
            UnitLocation::Node(node) => node.mutables.read().unwrap().units.contains(&unit),
            UnitLocation::Fleet(fleet) => fleet.get_daughters().contains(&unit),
            UnitLocation::Hangar(hangar) => {
                hangar.mutables.read().unwrap().contents.contains(&unit)
            }
        }
    }
    fn insert_unit(&self, unit: Unit) {
        match self {
            UnitLocation::Node(node) => node.mutables.write().unwrap().units.push(unit.clone()),
            UnitLocation::Fleet(fleet) => {
                fleet.mutables.write().unwrap().daughters.push(unit.clone())
            }
            UnitLocation::Hangar(hangar) => {
                hangar.mutables.write().unwrap().contents.push(unit.clone())
            }
        }
    }
    fn remove_unit(&self, unit: Unit) {
        match self {
            UnitLocation::Node(node) => node
                .mutables
                .write()
                .unwrap()
                .units
                .retain(|content| content != &unit),
            UnitLocation::Fleet(fleet) => fleet
                .mutables
                .write()
                .unwrap()
                .daughters
                .retain(|content| content != &unit),
            UnitLocation::Hangar(hangar) => hangar
                .mutables
                .write()
                .unwrap()
                .contents
                .retain(|content| content != &unit),
        }
    }
}

pub trait Mobility {
    fn new_unit(&self) -> Unit;
    fn get_ship(&self) -> Option<Arc<ShipInstance>>;
    fn get_fleet(&self) -> Option<Arc<FleetInstance>>;
    fn get_id(&self) -> u64;
    fn is_ship(&self) -> bool;
    fn get_location(&self) -> UnitLocation;
    fn check_location_coherency(&self);
    fn is_in_node(&self) -> bool {
        match self.get_location() {
            UnitLocation::Node(_) => true,
            _ => false,
        }
    }
    fn is_in_fleet(&self) -> bool {
        match self.get_location() {
            UnitLocation::Fleet(_) => true,
            _ => false,
        }
    }
    fn is_in_hangar(&self) -> bool {
        match self.get_location() {
            UnitLocation::Hangar(_) => true,
            _ => false,
        }
    }
    fn get_mother_node(&self) -> Arc<Node> {
        match self.get_location() {
            UnitLocation::Node(node) => node,
            UnitLocation::Fleet(fleet) => fleet.get_mother_node(),
            UnitLocation::Hangar(hangar) => hangar.mother.get_mother_node(),
        }
    }
    fn get_mother_fleet(&self) -> Option<Arc<FleetInstance>> {
        match self.get_location() {
            UnitLocation::Node(_) => None,
            UnitLocation::Fleet(fleet) => Some(fleet),
            UnitLocation::Hangar(hangar) => hangar.mother.get_mother_fleet(),
        }
    }
    fn get_allegiance(&self) -> Arc<Faction>;
    fn get_daughters(&self) -> Vec<Unit>;
    fn get_morale_scalar(&self) -> f32;
    fn get_character_strength_scalar(&self) -> f32;
    fn get_strength(&self, time: u64) -> u64;
    fn get_volume(&self) -> u64;
    fn get_ai(&self) -> NavAI;
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_ratio(&self, resource: Arc<Resource>) -> (u64, u64);
    fn get_shipclass_num(&self, shipclass: Arc<ShipClass>) -> u64;
    fn get_shipclass_supply(&self, shipclass: Arc<ShipClass>) -> u64;
    fn get_shipclass_demand(&self, shipclass: Arc<ShipClass>) -> u64;
    fn get_shipclass_demand_ratio(&self, shipclass: Arc<ShipClass>) -> (u64, u64);
    fn change_allegiance(&self, new_faction: Arc<Faction>);
    fn acyclicity_check(&self, location: UnitLocation) -> bool {
        match location.clone() {
            UnitLocation::Fleet(fleet) => {
                if fleet.get_id() == self.get_id() {
                    false
                } else {
                    self.acyclicity_check(fleet.get_location())
                }
            }
            UnitLocation::Hangar(hangar) => {
                let carrier = hangar.mother.clone();
                if carrier.get_id() == self.get_id() {
                    false
                } else {
                    self.acyclicity_check(carrier.get_location())
                }
            }
            _ => true,
        }
    }
    fn transfer(&self, destination: UnitLocation) -> bool;
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>>;
    //The ship and fleet implementations of this next method do slightly different things.
    //The ship version is used for gathering reinforcements, and assumes the ship can't make the move and its daughters will have to move independently.
    //It recurses down the tree, following that logic at every stage.
    //The fleet version is used to determine how much of the fleet can make a particular move.
    //Since granddaughters won't leave their mothers to accompany a fleet that leaves them beihnd, this version just checks the immediate daughters.
    //NOTE: It's maybe somewhat dangerous to do it like this? Could be confusing/unpredictable.
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit>;
    fn process_engines(&self, root: &Root, destination: Arc<Node>);
    fn navigate(&self, root: &Root, destinations: &Vec<Arc<Node>>) -> Option<Arc<Node>>;
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>>;
    fn maneuver(&self, root: &Root) -> Option<Arc<Node>>;
    fn try_defect(&self, root: &Root) -> Option<Vec<Arc<Node>>>;
    fn kill(&self);
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
    pub hangarvol: u64,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub defaultweapons: Option<HashMap<Arc<Resource>, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<Arc<HangarClass>>,
    pub engines: Vec<Arc<EngineClass>>,
    pub repairers: Vec<Arc<RepairerClass>>,
    pub factoryclasslist: Vec<Arc<FactoryClass>>,
    pub shipyardclasslist: Vec<Arc<ShipyardClass>>,
    pub aiclass: Arc<ShipAI>,
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
    //to avoid an infinite loop, this does not give the ship any hangars
    //we generate those in a later step with build_hangars
    fn instantiate(
        class: Arc<Self>,
        location: UnitLocation,
        faction: Arc<Faction>,
        root: &Root,
    ) -> ShipInstance {
        let index = root.unitcounter.fetch_add(1, atomic::Ordering::Relaxed);
        ShipInstance {
            id: index,
            visiblename: uuid::Uuid::new_v4().to_string(),
            class: class.clone(),
            mutables: RwLock::new(ShipInstanceMut {
                hull: class.basehull,
                visibility: class.visibility,
                stockpiles: class.stockpiles.clone(),
                efficiency: 1.0,
                hangars: Vec::new(),
                engines: class
                    .engines
                    .iter()
                    .map(|engineclass| EngineClass::instantiate(engineclass.clone()))
                    .collect(),
                movement_left: u64::MAX,
                repairers: class
                    .repairers
                    .iter()
                    .map(|repairerclass| RepairerClass::instantiate(repairerclass.clone()))
                    .collect(),
                factoryinstancelist: class
                    .factoryclasslist
                    .iter()
                    .map(|factoryclass| FactoryClass::instantiate(factoryclass.clone()))
                    .collect(),
                shipyardinstancelist: class
                    .shipyardclasslist
                    .iter()
                    .map(|shipyardclass| {
                        ShipyardClass::instantiate(shipyardclass.clone(), &root.shipclasses)
                    })
                    .collect(),
                location,
                allegiance: faction,
                objectives: Vec::new(),
                aiclass: class.aiclass.clone(),
            }),
        }
    }
    //NOTE: having this be a method feels a little odd when instantiate isn't one
    pub fn build_hangars(
        &self,
        ship: Arc<ShipInstance>,
        shipclasses: &Vec<Arc<ShipClass>>,
        counter: &Arc<AtomicU64>,
    ) {
        let hangars: Vec<_> = self
            .hangars
            .iter()
            .map(|hangarclass| {
                Arc::new(HangarClass::instantiate(
                    hangarclass.clone(),
                    ship.clone(),
                    shipclasses,
                    counter,
                ))
            })
            .collect();
        ship.mutables.write().unwrap().hangars = hangars;
    }
}

#[derive(Debug, Clone)]
pub struct ShipInstanceMut {
    pub hull: u64, //how many hitpoints the ship has
    pub visibility: bool,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<Arc<HangarInstance>>,
    pub engines: Vec<EngineInstance>,
    pub movement_left: u64, //starts at one trillion each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<RepairerInstance>,
    pub factoryinstancelist: Vec<FactoryInstance>,
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub location: UnitLocation, //where the ship is -- a node if it's unaffiliated, a fleet if it's in one
    pub allegiance: Arc<Faction>, //which faction this ship belongs to
    pub objectives: Vec<ObjectiveFlavor>,
    pub aiclass: Arc<ShipAI>,
}

#[derive(Debug)]
pub struct ShipInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: Arc<ShipClass>, //which class of ship this is
    pub mutables: RwLock<ShipInstanceMut>,
}

impl PartialEq for ShipInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipInstance {}

impl Ord for ShipInstance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipInstance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Mobility for Arc<ShipInstance> {
    fn new_unit(&self) -> Unit {
        Unit::Ship(self.clone())
    }
    fn get_ship(&self) -> Option<Arc<ShipInstance>> {
        Some(self.clone())
    }
    fn get_fleet(&self) -> Option<Arc<FleetInstance>> {
        None
    }
    fn get_id(&self) -> u64 {
        self.id
    }
    fn is_ship(&self) -> bool {
        true
    }
    fn get_location(&self) -> UnitLocation {
        self.mutables.read().unwrap().location.clone()
    }
    fn check_location_coherency(&self) {
        let mother = self.get_location();
        let sisters: Vec<_> = match mother.clone() {
            UnitLocation::Node(node) => node
                .mutables
                .read()
                .unwrap()
                .units
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Fleet(fleet) => fleet
                .mutables
                .read()
                .unwrap()
                .daughters
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Hangar(hangar) => hangar
                .mutables
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
        };
        assert_eq!(
            sisters
                .iter()
                .filter(|sister| sister.get_id() == self.id)
                .count(),
            1
        );
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        self.mutables.read().unwrap().allegiance.clone()
    }
    fn get_daughters(&self) -> Vec<Unit> {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| {
                hangar
                    .mutables
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .map(|unit| {
                        let mut vec = unit.get_daughters();
                        vec.insert(0, unit.clone());
                        vec
                    })
                    .collect::<Vec<Vec<Unit>>>()
            })
            .flatten()
            .flatten()
            .collect()
    }
    //NOTE: Dummied out until morale system exists.
    fn get_morale_scalar(&self) -> f32 {
        1.0
    }
    //NOTE: Dummied out until characters exist.
    fn get_character_strength_scalar(&self) -> f32 {
        1.0
    }
    fn get_strength(&self, time: u64) -> u64 {
        let mutables = self.mutables.read().unwrap();
        let daughter_strength = mutables
            .hangars
            .iter()
            .map(|hangar| hangar.get_strength(time))
            .sum::<u64>();
        let objective_strength: f32 = mutables
            .objectives
            .iter()
            .map(|of| of.get_scalars().strengthscalar)
            .product();
        (self.class.basestrength as f32
            * (mutables.hull as f32 / self.class.basehull as f32)
            * self.get_character_strength_scalar()
            * objective_strength) as u64
            + daughter_strength
    }
    fn get_volume(&self) -> u64 {
        self.class.hangarvol
    }
    fn get_ai(&self) -> NavAI {
        let ai = &self.mutables.read().unwrap().aiclass;
        NavAI {
            ship_attract_specific: ai.ship_attract_specific.clone(),
            ship_attract_generic: ai.ship_attract_generic.clone(),
            ship_cargo_attract: ai.ship_cargo_attract.clone(),
            resource_attract: ai.resource_attract.clone(),
        }
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>()
            + mutables
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_supply_total(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagate)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
            + mutables
                .engines
                .iter()
                .map(|e| e.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .repairers
                .iter()
                .map(|r| r.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
            + mutables
                .shipyardinstancelist
                .iter()
                .map(|s| s.get_resource_demand_total(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand_ratio(&self, resource: Arc<Resource>) -> (u64, u64) {
        let mutables = self.mutables.read().unwrap();
        let demand_total = mutables
            .stockpiles
            .iter()
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>();
        let target_total = mutables.stockpiles.iter().map(|sp| sp.target).sum::<u64>();
        assert!(demand_total < target_total);
        (demand_total, target_total)
    }
    fn get_shipclass_num(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| hangar.get_shipclass_num(shipclass.clone()))
            .sum::<u64>()
    }
    fn get_shipclass_supply(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| hangar.get_shipclass_supply(shipclass.clone()))
            .sum::<u64>()
            + if self.class == shipclass {
                self.class.hangarvol
            } else {
                0
            }
    }
    fn get_shipclass_demand(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| hangar.get_shipclass_demand(shipclass.clone()))
            .sum::<u64>()
    }
    fn get_shipclass_demand_ratio(&self, shipclass: Arc<ShipClass>) -> (u64, u64) {
        let mutables = self.mutables.read().unwrap();
        let demand_total = mutables
            .hangars
            .iter()
            .map(|hangar| hangar.get_shipclass_demand(shipclass.clone()))
            .sum::<u64>();
        let target_total = mutables
            .hangars
            .iter()
            .map(|hangar| hangar.class.target)
            .sum::<u64>();
        assert!(demand_total < target_total);
        (demand_total, target_total)
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        self.mutables.write().unwrap().allegiance = new_faction.clone();
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.change_allegiance(new_faction.clone()));
    }
    //NOTE: Figure out if making new boxes here will actually work. If not, trait transferability on unit time, probably.
    fn transfer(&self, destination: UnitLocation) -> bool {
        let source = self.get_location();
        if source.check_remove(self.new_unit())
            && destination.check_insert(self.new_unit())
            && self.acyclicity_check(destination.clone())
        {
            source.remove_unit(self.new_unit());
            //NOTE: Make sure cloning destination here clones the arc rather than cloning the thing inside the arc
            self.mutables.write().unwrap().location = destination.clone();
            destination.insert_unit(self.new_unit());
            true
        } else {
            false
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        let location = self.get_mother_node();
        let mutables = self.mutables.read().unwrap();
        if mutables.movement_left > 0 {
            if let Some((viable, _speed)) = mutables
                .engines
                .iter()
                .find_map(|e| e.check_engine(root, location.clone(), destinations))
            {
                Some(viable)
            } else {
                None
            }
        } else {
            None
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .map(|hangar| {
                let (active, passive): (Vec<Unit>, Vec<Unit>) = hangar
                    .mutables
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .cloned()
                    .partition(|ship| {
                        ship.destinations_check(root, &vec![destination.clone()])
                            .is_some()
                    });
                let active_daughters = active
                    .iter()
                    .map(|unit| {
                        let mut active_daughters_mut = unit.get_daughters();
                        active_daughters_mut.insert(0, unit.clone());
                        active_daughters_mut
                    })
                    .collect::<Vec<Vec<Unit>>>();
                let passive_daughters = passive
                    .iter()
                    .map(|unit| {
                        let mut passive_daughters_mut =
                            unit.get_traversal_checked_daughters(root, destination.clone());
                        passive_daughters_mut.insert(0, unit.clone());
                        passive_daughters_mut
                    })
                    .collect::<Vec<Vec<Unit>>>();
                vec![active_daughters, passive_daughters]
            })
            .flatten()
            .flatten()
            .flatten()
            .collect()
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        let location = self.get_mother_node();
        let mut mutables = self.mutables.write().unwrap();
        match mutables.movement_left > 0 {
            true => {
                if let Some(speed) = mutables
                    .engines
                    .iter_mut()
                    .find_map(|e| e.process_engine(root, location.clone(), destination.clone()))
                {
                    mutables.movement_left =
                        mutables.movement_left.saturating_sub(u64::MAX / speed);
                } else {
                    panic!();
                }
            }
            false => {
                panic!();
            }
        }
    }
    fn navigate(
        //used for ships which are operating independently
        //this method determines which of the current node's neighbors is most desirable
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Arc<Node>> {
        let location: Arc<Node> = self.get_mother_node();
        let mutables = self.mutables.read().unwrap();
        //we iterate over the destinations to determine which neighbor is most desirable
        destinations
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .max_by_key(|node| {
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                let resource_demand_value: f32 = mutables
                    .aiclass
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        let demand = root.globalsalience.resourcesalience.read().unwrap()
                            [mutables.allegiance.id][resource.id][node.id][0];
                        let supply = root.globalsalience.resourcesalience.read().unwrap()
                            [mutables.allegiance.id][resource.id][node.id][1];
                        //let cargo = self.stockpiles.iter().map(|x|)
                        (demand - supply)
                            * self.get_resource_supply(resource.clone()) as f32
                            * scalar
                    })
                    .sum();
                let resource_supply_value: f32 = mutables
                    .aiclass
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        //we index into the salience map by resource and then by node
                        //to determine how much supply there is in this node for each resource the subject ship wants
                        //NOTE: Previously, we got demand by indexing by nodeid, not position.
                        //I believe using the ship's current position to calculate demand
                        //will eliminate a pathology and produce more correct gradient-following behavior.
                        let demand = root.globalsalience.resourcesalience.read().unwrap()
                            [mutables.allegiance.id][resource.id][node.id][0];
                        let supply = root.globalsalience.resourcesalience.read().unwrap()
                            [mutables.allegiance.id][resource.id][node.id][1];
                        let (demand_total, target_total) =
                            self.get_resource_demand_ratio(resource.clone());
                        supply * demand * (demand_total as f32 / target_total as f32) * scalar
                    })
                    .sum();
                let shipclass_demand_value: f32 = mutables
                    .aiclass
                    .ship_cargo_attract
                    .iter()
                    .map(|(shipclassid, scalar)| {
                        let attractive_shipclass = root.shipclasses[shipclassid.index].clone();
                        let demand = root.globalsalience.shipclasssalience.read().unwrap()
                            [mutables.allegiance.id][shipclassid.index][node.id][0];
                        let supply = root.globalsalience.shipclasssalience.read().unwrap()
                            [mutables.allegiance.id][shipclassid.index][node.id][1];
                        (demand - supply)
                            * (self.get_shipclass_num(attractive_shipclass.clone()) as f32
                                * attractive_shipclass.hangarvol as f32)
                            * scalar
                    })
                    .sum();
                //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
                let shipclass_supply_value: f32 = mutables
                    .aiclass
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
                        let demand = root.globalsalience.shipclasssalience.read().unwrap()
                            [mutables.allegiance.id][shipclassid.index][location.id][0];
                        let supply = root.globalsalience.shipclasssalience.read().unwrap()
                            [mutables.allegiance.id][shipclassid.index][node.id][1];
                        let (demand_total, target_total) =
                            self.get_shipclass_demand_ratio(attractive_shipclass.clone());
                        supply * demand * (demand_total as f32 / target_total as f32) * scalar
                    })
                    .sum();
                //this checks how much demand there is in the node for ships of the subject ship's class
                let ship_value_specific: f32 =
                    root.globalsalience.shipclasssalience.read().unwrap()[mutables.allegiance.id]
                        [self.class.id][node.id][0]
                        * mutables.aiclass.ship_attract_specific;
                //oh, THIS is why we needed the placeholder ship class
                //this checks how much demand there is in the node for ships in general
                let ship_value_generic: f32 = root.globalsalience.shipclasssalience.read().unwrap()
                    [mutables.allegiance.id][0][node.id][0]
                    * mutables.aiclass.ship_attract_generic;

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
            .cloned()
        //if this doesn't work for some reason, we return None
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        self.process_engines(root, destination.clone());
        self.transfer(UnitLocation::Node(destination.clone()));
        Some(destination)
    }
    //this moves a ship across one edge so long as it has a functioning engine, draining fuel from the engines it uses
    //in turn processing, we'll need to repeat traversal as long as it continues returning true
    fn maneuver(&self, root: &Root) -> Option<Arc<Node>> {
        let location = self.get_mother_node();
        let empty = Vec::new();
        let neighbors = root.neighbors.get(&location).unwrap_or(&empty);
        match self.destinations_check(root, &neighbors) {
            Some(destinations) => {
                let destination_option = self.navigate(root, &destinations);
                match destination_option.clone() {
                    Some(destination) => {
                        self.traverse(root, destination);
                    }
                    None => {}
                }
                destination_option
            }
            None => None,
        }
    }
    //Checks whether the shipinstance will defect this turn; if it will, makes the ship defect and returns the node the ship is in afterward
    //so that that node can be checked for engagements
    fn try_defect(&self, root: &Root) -> Option<Vec<Arc<Node>>> {
        let location = self.get_mother_node();
        let mut mutables = self.mutables.write().unwrap();
        //NOTE: Test whether this gives us numbers that make sense.
        let local_threat_ratio: f32 = self
            .class
            .defectchance
            .iter()
            .map(|(faction, _)| {
                root.globalsalience.factionsalience.read().unwrap()[mutables.allegiance.id]
                    [faction.id][location.id][0]
            })
            .sum();
        //if defectchance only contains data for one faction
        //then either it's the faction it's currently part of, in which case we have no data on its chance of joining other factions
        //or it's not, in which case we have no data on its chance of defecting from its current faction
        //so either way we treat it as incapable of defecting
        let defect_probability = if self.class.defectchance.len() > 1 {
            ((local_threat_ratio * self.class.defectchance
                .get(&mutables.allegiance)
                .unwrap_or(&(0.0, 0.0))
                .0)
                //NOTE: I *think* that clamp will resolve things properly if we end up dividing by zero -- it'll set the probability to 1 -- but it's hard to be sure
                /self.get_morale_scalar())
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
                            * root.globalsalience.factionsalience.read().unwrap()[faction.id]
                                [faction.id][location.id][0]),
                    )
                })
                .collect();
            let new_faction: Arc<Faction> = new_faction_probabilities
                .choose_weighted(&mut rng, |(_, prob)| prob.clone())
                .unwrap()
                .0
                .clone();
            mutables.allegiance = new_faction.clone();
            //NOTE: This should take more things into account probably
            let escapes = rng.gen_bool(self.class.defectescapescalar.clamp(0.0, 1.0) as f64);
            if escapes {
                let destinations_option =
                    self.destinations_check(root, root.neighbors.get(&location).unwrap());
                match destinations_option.clone() {
                    Some(destinations) => {
                        let destination = destinations
                            .iter()
                            .max_by_key(|node| {
                                root.globalsalience.factionsalience.read().unwrap()[new_faction.id]
                                    [new_faction.id][node.id][0]
                                    as i64
                            })
                            .unwrap()
                            .clone();
                        self.traverse(root, destination.clone());
                        Some(vec![destination])
                    }
                    None => {
                        self.transfer(UnitLocation::Node(location.clone()));
                        Some(vec![location])
                    }
                }
            } else {
                self.transfer(UnitLocation::Node(location.clone()));
                Some(vec![location])
            }
        } else {
            None
        }
    }
    fn kill(&self) {
        self.mutables.write().unwrap().hull = 0;
        self.get_daughters().iter().for_each(|ship| ship.kill());
    }
}

impl ShipInstance {
    pub fn repair(&self, per_engagement: bool) {
        let ship = self.class.visiblename.clone();
        //dbg!(ship);
        let mut mutables: RwLockWriteGuard<ShipInstanceMut> = self.mutables.write().unwrap();
        let current_hull = mutables.hull;
        if current_hull < self.class.basehull
            || mutables.engines.iter().any(|e| e.health < e.basehealth)
        {
            let [(hull_repair_points, hull_repair_factor), (engine_repair_points, engine_repair_factor)]: [(i64, f32); 2] = mutables
                .repairers
                .iter_mut()
                .filter(|rp| rp.per_engagement == per_engagement)
                .filter(|rp| rp.get_state() == FactoryState::Active)
                .map(|rp| {
                    rp.process();
                    [
                        (rp.repair_points, rp.repair_factor),
                        (rp.engine_repair_points, rp.engine_repair_factor),
                    ]
                })
                .fold([(0, 0.0); 2], |a, b| {
                    [
                        (a[0].0 + b[0].0, a[0].1 + b[0].1),
                        (a[1].0 + b[1].0, a[1].1 + b[1].1),
                    ]
                });
            mutables.hull = (current_hull as i64
                + hull_repair_points
                + (self.class.basehull as f32 * hull_repair_factor) as i64)
                .clamp(0, self.class.basehull as i64) as u64;
            mutables
                .engines
                .iter_mut()
                .filter(|e| e.health.is_some())
                .for_each(|e| {
                    (e.health.unwrap() as i64
                        + engine_repair_points
                        + (e.basehealth.unwrap() as f32 * engine_repair_factor) as i64)
                        .clamp(0, e.basehealth.unwrap() as i64) as u64;
                })
        }
    }
    pub fn process_factories(&self) {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        mutables
            .factoryinstancelist
            .iter_mut()
            .for_each(|f| f.process(efficiency));
    }
    pub fn process_shipyards(&self) {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        mutables
            .shipyardinstancelist
            .iter_mut()
            .for_each(|sy| sy.process(efficiency));
    }
    pub fn plan_ships(
        &self,
        shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)> {
        let mut mutables = self.mutables.write().unwrap();
        let efficiency = mutables.efficiency;
        let location = mutables.location.clone();
        let allegiance = mutables.allegiance.clone();
        mutables
            .shipyardinstancelist
            .iter_mut()
            .map(|shipyard| {
                let ship_plans = shipyard.plan_ships(efficiency, shipclasses);
                //here we take the list of ships for a specific shipyard and tag them with the location and allegiance they should have when they're built
                ship_plans
                    .iter()
                    .map(|ship_plan| (ship_plan.clone(), location.clone(), allegiance.clone()))
                    // <^>>(
                    .collect::<Vec<_>>()
            })
            //we flatten the collection of vecs corresponding to individual shipyards, because we just want to create all the ships and don't care who created them
            .flatten()
            .collect::<Vec<_>>()
    }
    pub fn reset_movement(&self) {
        self.mutables.write().unwrap().movement_left = u64::MAX;
    }
}

#[derive(Debug, Clone)]
pub struct FleetClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub strengthmod: (f32, u64),
    pub fleetconfig: HashMap<Arc<ShipClass>, u64>,
    pub defectchance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub defectescapescalar: f32,
    pub navquorum: f32,
    pub disbandthreshold: f32,
}

impl PartialEq for FleetClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FleetClass {}

impl Ord for FleetClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for FleetClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for FleetClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl FleetClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.fleetconfig
            .iter()
            .map(|(shipclass, v)| shipclass.get_ideal_strength(root) * v)
            .sum()
    }
    pub fn instantiate(
        class: Arc<Self>,
        location: Arc<Node>,
        faction: Arc<Faction>,
        root: &Root,
    ) -> FleetInstance {
        let index = root.unitcounter.fetch_add(1, atomic::Ordering::Relaxed);
        FleetInstance {
            id: index,
            visiblename: uuid::Uuid::new_v4().to_string(),
            class: class.clone(),
            idealstrength: class.get_ideal_strength(root),
            mutables: RwLock::new(FleetInstanceMut {
                visibility: class.visibility,
                location: UnitLocation::Node(location),
                daughters: Vec::new(),
                allegiance: faction,
                objectives: Vec::new(),
                phantom: true,
            }),
        }
    }
}

//NOTE: At present, this is an exact copy of ShipAI except that it doesn't have an ID, because it's always calculated on the fly.
#[derive(Debug, Clone)]
pub struct NavAI {
    pub ship_attract_specific: f32,
    pub ship_attract_generic: f32,
    pub ship_cargo_attract: HashMap<ShipClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>,
}

#[derive(Debug, Clone)]
pub struct FleetInstanceMut {
    visibility: bool,
    location: UnitLocation,
    daughters: Vec<Unit>,
    allegiance: Arc<Faction>,
    objectives: Vec<ObjectiveFlavor>,
    phantom: bool,
}

#[derive(Debug)]
pub struct FleetInstance {
    id: u64,
    visiblename: String,
    class: Arc<FleetClass>,
    idealstrength: u64,
    mutables: RwLock<FleetInstanceMut>,
}

impl PartialEq for FleetInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FleetInstance {}

impl Ord for FleetInstance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for FleetInstance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for FleetInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Mobility for Arc<FleetInstance> {
    fn new_unit(&self) -> Unit {
        Unit::Fleet(self.clone())
    }
    fn get_ship(&self) -> Option<Arc<ShipInstance>> {
        None
    }
    fn get_fleet(&self) -> Option<Arc<FleetInstance>> {
        Some(self.clone())
    }
    fn get_id(&self) -> u64 {
        self.id
    }
    fn is_ship(&self) -> bool {
        false
    }
    fn get_location(&self) -> UnitLocation {
        self.mutables.read().unwrap().location.clone()
    }
    fn check_location_coherency(&self) {
        let mother = self.get_location();
        let sisters: Vec<_> = match mother.clone() {
            UnitLocation::Node(node) => node
                .mutables
                .read()
                .unwrap()
                .units
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Fleet(fleet) => fleet
                .mutables
                .read()
                .unwrap()
                .daughters
                .iter()
                .cloned()
                .collect(),
            UnitLocation::Hangar(hangar) => hangar
                .mutables
                .read()
                .unwrap()
                .contents
                .iter()
                .cloned()
                .collect(),
        };
        assert_eq!(
            sisters
                .iter()
                .filter(|sister| sister.get_id() == self.id)
                .count(),
            1
        );
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        self.mutables.read().unwrap().allegiance.clone()
    }
    fn get_daughters(&self) -> Vec<Unit> {
        self.mutables
            .read()
            .unwrap()
            .daughters
            .iter()
            .cloned()
            .collect()
    }
    fn get_morale_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_morale_scalar())
            .product()
    }
    fn get_character_strength_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_character_strength_scalar())
            .product()
    }
    fn get_strength(&self, time: u64) -> u64 {
        let (factor, additive) = self.class.strengthmod;
        let sum = self
            .get_daughters()
            .iter()
            .map(|daughter| daughter.get_strength(time))
            .sum::<u64>();
        (sum as f32 * factor) as u64 + additive
    }
    fn get_volume(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_volume())
            .sum()
    }
    fn get_ai(&self) -> NavAI {
        self.get_daughters().iter().fold(
            NavAI {
                ship_attract_specific: 1.0,
                ship_attract_generic: 1.0,
                ship_cargo_attract: HashMap::new(),
                resource_attract: HashMap::new(),
            },
            |mut acc, ship| {
                let sub_ai = ship.get_ai();
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
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_supply(resource.clone()))
            .sum()
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_resource_demand_ratio(&self, resource: Arc<Resource>) -> (u64, u64) {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand_ratio(resource.clone()))
            .fold((0, 0), |acc, rhs| (acc.0 + rhs.0, acc.1 + rhs.1))
    }
    fn get_shipclass_num(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.get_daughters()
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.class == shipclass)
            .collect::<Vec<_>>()
            .len() as u64
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_shipclass_num(shipclass.clone()))
                .sum::<u64>()
    }
    fn get_shipclass_supply(&self, shipclass: Arc<ShipClass>) -> u64 {
        self.get_daughters()
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.class == shipclass)
            .map(|_| shipclass.hangarvol)
            .sum::<u64>()
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_shipclass_supply(shipclass.clone()))
                .sum::<u64>()
    }
    fn get_shipclass_demand(&self, shipclass: Arc<ShipClass>) -> u64 {
        let ideal_num = self.class.fleetconfig.get(&shipclass).unwrap_or(&0);
        ((ideal_num.saturating_sub(self.get_shipclass_num(shipclass.clone())))
            * shipclass.hangarvol)
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_shipclass_demand(shipclass.clone()))
                .sum::<u64>()
    }
    fn get_shipclass_demand_ratio(&self, shipclass: Arc<ShipClass>) -> (u64, u64) {
        let self_demand_total = self.get_shipclass_demand(shipclass.clone());
        let self_target_total = self
            .class
            .fleetconfig
            .iter()
            .map(|(class, num)| class.hangarvol * num)
            .sum::<u64>();
        let (daughter_demand_total, daughter_target_total) = self
            .get_daughters()
            .iter()
            .map(|daughter| daughter.get_shipclass_demand_ratio(shipclass.clone()))
            .fold((0, 0), |acc, rhs| (acc.0 + rhs.0, acc.1 + rhs.1));
        (
            self_demand_total + daughter_demand_total,
            self_target_total + daughter_target_total,
        )
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        self.mutables.write().unwrap().allegiance = new_faction.clone();
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.change_allegiance(new_faction.clone()));
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        match self.mutables.read().unwrap().location.clone() {
            UnitLocation::Node(node) => {
                //NOTE: Make sure cloning destination here clones the arc rather than cloning the thing inside the arc
                match destination.clone() {
                    UnitLocation::Node(destnode) => {
                        node.mutables
                            .write()
                            .unwrap()
                            .units
                            .retain(|unit| unit.get_id() != self.id);
                        self.mutables.write().unwrap().location = destination.clone();
                        destnode
                            .mutables
                            .write()
                            .unwrap()
                            .units
                            .push(self.new_unit());
                        true
                    }
                    UnitLocation::Fleet(_) => false,
                    UnitLocation::Hangar(_) => false,
                }
            }
            UnitLocation::Fleet(_) => false,
            UnitLocation::Hangar(_) => false,
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        let viable: Vec<_> = destinations
            .iter()
            .filter(|node| {
                self.get_traversal_checked_daughters(root, node.clone().clone())
                    .len()
                    > 0
            })
            .cloned()
            .collect();
        if !viable.is_empty() {
            Some(viable)
        } else {
            None
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        let daughters = self.get_daughters();
        let (passed_ships, failed_ships): (Vec<Unit>, Vec<Unit>) =
            daughters.iter().cloned().partition(|daughter| {
                daughter
                    .destinations_check(root, &vec![destination.clone()])
                    .is_some()
            });
        //we see what fraction of the fleet's strength is unable to make the jump
        //by checking strength of failed ships, and then all daughters
        //we don't just call get_strength on the fleet itself
        //if we did, the fleet's strength modifiers would be counted only toward its total
        let failed_strength = failed_ships
            .iter()
            .map(|ship| ship.get_strength(root.config.battlescalars.avg_duration) as f32)
            .sum::<f32>();
        let total_strength = daughters
            .iter()
            .map(|daughter| daughter.get_strength(root.config.battlescalars.avg_duration) as f32)
            .sum::<f32>();
        if (failed_strength / total_strength) < (1.0 - self.class.navquorum) {
            passed_ships
        } else {
            Vec::new()
        }
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.process_engines(root, destination.clone()));
    }
    fn navigate(&self, root: &Root, destinations: &Vec<Arc<Node>>) -> Option<Arc<Node>> {
        //we iterate over the destinations to determine which neighbor is most desirable
        destinations
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .max_by_key(|node| {
                let ai = self.get_ai();
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                let resource_demand_value: f32 = ai
                    .resource_attract
                    .iter()
                    .map(|(resource, scalar)| {
                        let demand = root.globalsalience.resourcesalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][resource.id][node.id][0];
                        let supply = root.globalsalience.resourcesalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][resource.id][node.id][1];
                        (demand - supply)
                            * self.get_resource_supply(resource.clone()) as f32
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
                        let demand = root.globalsalience.resourcesalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][resource.id]
                            [self.get_mother_node().id][0];
                        let supply = root.globalsalience.resourcesalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][resource.id][node.id][1];
                        let (demand_total, target_total) =
                            self.get_resource_demand_ratio(resource.clone());
                        supply * demand * (demand_total as f32 / target_total as f32) * scalar
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
                        let demand = root.globalsalience.shipclasssalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][shipclassid.index]
                            [node.id][0];
                        let supply = root.globalsalience.shipclasssalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][shipclassid.index]
                            [node.id][1];
                        (demand - supply)
                            * (self.get_shipclass_num(attractive_shipclass.clone()) as f32
                                * attractive_shipclass.hangarvol as f32)
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
                        let demand = root.globalsalience.shipclasssalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][shipclassid.index]
                            [self.get_mother_node().id][0];
                        let supply = root.globalsalience.shipclasssalience.read().unwrap()
                            [self.mutables.read().unwrap().allegiance.id][shipclassid.index]
                            [node.id][1];
                        let (demand_total, target_total) =
                            self.get_shipclass_demand_ratio(attractive_shipclass.clone());
                        supply * demand * (demand_total as f32 / target_total as f32) * scalar
                    })
                    .sum();
                //NOTE: Here in the fleet implementation, I have removed the component that determines how much demand there is for ships of this ship's class.
                //In theory we could implement this by adding up the demand for the ships of the daughters' various classes, scaled according to their individual
                //ship_attract_specifics, but right now I'm not going to bother.

                //this checks how much demand there is in the node for ships in general
                let ship_value_generic: f32 = root.globalsalience.shipclasssalience.read().unwrap()
                    [self.mutables.read().unwrap().allegiance.id][0][node.id][0]
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
            .cloned()
        //if this doesn't work for some reason, we return None
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        let daughters = self.get_daughters();
        let valid = self.get_traversal_checked_daughters(root, destination.clone());
        if valid.len() > 0 {
            let left_behind = daughters
                .iter()
                .filter(|daughter| !valid.contains(daughter))
                .collect::<Vec<_>>();
            left_behind.iter().for_each(|expelled| {
                expelled.transfer(UnitLocation::Node(self.get_mother_node()));
            });
            self.process_engines(root, destination.clone());
            self.transfer(UnitLocation::Node(destination.clone()));
            Some(destination)
        } else {
            None
        }
    }
    fn maneuver(&self, root: &Root) -> Option<Arc<Node>> {
        let neighbors = root
            .neighbors
            .get(&self.get_mother_node())
            .unwrap_or(&Vec::new())
            .clone();
        if let Some(destinations) = self.destinations_check(root, &neighbors) {
            let destination_option = self.navigate(root, &destinations);
            match destination_option.clone() {
                Some(destination) => {
                    self.traverse(root, destination.clone());
                }
                None => {}
            }
            destination_option
        } else {
            None
        }
    }
    //Checks whether the fleetinstance will defect this turn; if it will, makes the ship defect and returns the node the ship is in afterward
    //so that that node can be checked for engagements
    fn try_defect(&self, root: &Root) -> Option<Vec<Arc<Node>>> {
        //if defectchance only contains data for one faction
        //then either it's the faction it's currently part of, in which case we have no data on its chance of joining other factions
        //or it's not, in which case we have no data on its chance of defecting from its current faction
        //so either way we treat it as incapable of defecting
        //NOTE: Test whether this gives us numbers that make sense.
        let local_threat_ratio: f32 = self
            .class
            .defectchance
            .iter()
            .map(|(faction, _)| {
                root.globalsalience.factionsalience.read().unwrap()
                    [self.mutables.read().unwrap().allegiance.id][faction.id]
                    [self.get_mother_node().id][0]
            })
            .sum();
        let defect_probability = if self.class.defectchance.len() > 1 {
            ((local_threat_ratio * self.class.defectchance
                .get(&self.mutables.read().unwrap().allegiance)
                .unwrap_or(&(0.0, 0.0))
                .0)
                //NOTE: I *think* that clamp will resolve things properly if we end up dividing by zero -- it'll set the probability to 1 -- but it's hard to be sure
                /self.get_morale_scalar())
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
                            * root.globalsalience.factionsalience.read().unwrap()[faction.id]
                                [faction.id][self.get_mother_node().id][0]),
                    )
                })
                .collect();
            let new_faction: Arc<Faction> = new_faction_probabilities
                .choose_weighted(&mut rng, |(_, prob)| prob.clone())
                .unwrap()
                .0
                .clone();
            self.change_allegiance(new_faction.clone());
            let escapes = rng.gen_bool(self.class.defectescapescalar.clamp(0.0, 1.0) as f64);
            if escapes {
                let neighbors = root.neighbors.get(&self.get_mother_node()).unwrap().clone();
                let destinations_option = self.destinations_check(root, &neighbors);
                match destinations_option {
                    Some(destinations) => {
                        let destination = destinations
                            .iter()
                            .max_by_key(|node| {
                                root.globalsalience.factionsalience.read().unwrap()[new_faction.id]
                                    [new_faction.id][node.id][0]
                                    as i64
                            })
                            .unwrap();
                        self.traverse(root, destination.clone());
                        Some(vec![self.get_mother_node(), destination.clone()])
                    }
                    None => Some(vec![self.get_mother_node()]),
                }
            } else {
                Some(vec![self.get_mother_node()])
            }
        } else {
            None
        }
    }
    fn kill(&self) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.kill());
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Unit {
    Ship(Arc<ShipInstance>),
    Fleet(Arc<FleetInstance>),
}

impl Mobility for Unit {
    fn new_unit(&self) -> Unit {
        self.clone()
    }
    fn get_ship(&self) -> Option<Arc<ShipInstance>> {
        match self {
            Unit::Ship(ship) => Some(ship.clone()),
            _ => None,
        }
    }
    fn get_fleet(&self) -> Option<Arc<FleetInstance>> {
        match self {
            Unit::Fleet(fleet) => Some(fleet.clone()),
            _ => None,
        }
    }
    fn get_id(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_id(),
            Unit::Fleet(fleet) => fleet.get_id(),
        }
    }
    fn is_ship(&self) -> bool {
        match self {
            Unit::Ship(ship) => ship.is_ship(),
            Unit::Fleet(fleet) => fleet.is_ship(),
        }
    }
    fn get_location(&self) -> UnitLocation {
        match self {
            Unit::Ship(ship) => ship.get_location(),
            Unit::Fleet(fleet) => fleet.get_location(),
        }
    }
    fn check_location_coherency(&self) {
        match self {
            Unit::Ship(ship) => ship.check_location_coherency(),
            Unit::Fleet(fleet) => fleet.check_location_coherency(),
        }
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        match self {
            Unit::Ship(ship) => ship.get_allegiance(),
            Unit::Fleet(fleet) => fleet.get_allegiance(),
        }
    }
    fn get_daughters(&self) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_daughters(),
            Unit::Fleet(fleet) => fleet.get_daughters(),
        }
    }
    fn get_morale_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_morale_scalar(),
            Unit::Fleet(fleet) => fleet.get_morale_scalar(),
        }
    }
    fn get_character_strength_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_character_strength_scalar(),
            Unit::Fleet(fleet) => fleet.get_character_strength_scalar(),
        }
    }
    fn get_strength(&self, time: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength(time),
            Unit::Fleet(fleet) => fleet.get_strength(time),
        }
    }
    fn get_volume(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_volume(),
            Unit::Fleet(fleet) => fleet.get_volume(),
        }
    }
    fn get_ai(&self) -> NavAI {
        match self {
            Unit::Ship(ship) => ship.get_ai(),
            Unit::Fleet(fleet) => fleet.get_ai(),
        }
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_supply(resource),
            Unit::Fleet(fleet) => fleet.get_resource_supply(resource),
        }
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand(resource),
            Unit::Fleet(fleet) => fleet.get_resource_demand(resource),
        }
    }
    fn get_resource_demand_ratio(&self, resource: Arc<Resource>) -> (u64, u64) {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand_ratio(resource),
            Unit::Fleet(fleet) => fleet.get_resource_demand_ratio(resource),
        }
    }
    fn get_shipclass_num(&self, shipclass: Arc<ShipClass>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_shipclass_num(shipclass),
            Unit::Fleet(fleet) => fleet.get_shipclass_num(shipclass),
        }
    }
    fn get_shipclass_supply(&self, shipclass: Arc<ShipClass>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_shipclass_supply(shipclass),
            Unit::Fleet(fleet) => fleet.get_shipclass_supply(shipclass),
        }
    }
    fn get_shipclass_demand(&self, shipclass: Arc<ShipClass>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_shipclass_demand(shipclass),
            Unit::Fleet(fleet) => fleet.get_shipclass_demand(shipclass),
        }
    }
    fn get_shipclass_demand_ratio(&self, shipclass: Arc<ShipClass>) -> (u64, u64) {
        match self {
            Unit::Ship(ship) => ship.get_shipclass_demand_ratio(shipclass),
            Unit::Fleet(fleet) => fleet.get_shipclass_demand_ratio(shipclass),
        }
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        match self {
            Unit::Ship(ship) => ship.change_allegiance(new_faction),
            Unit::Fleet(fleet) => fleet.change_allegiance(new_faction),
        }
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        match self {
            Unit::Ship(ship) => ship.transfer(destination),
            Unit::Fleet(fleet) => fleet.transfer(destination),
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        match self {
            Unit::Ship(ship) => ship.destinations_check(root, destinations),
            Unit::Fleet(fleet) => fleet.destinations_check(root, destinations),
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_traversal_checked_daughters(root, destination),
            Unit::Fleet(fleet) => fleet.get_traversal_checked_daughters(root, destination),
        }
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        match self {
            Unit::Ship(ship) => ship.process_engines(root, destination),
            Unit::Fleet(fleet) => fleet.process_engines(root, destination),
        }
    }
    fn navigate(&self, root: &Root, destinations: &Vec<Arc<Node>>) -> Option<Arc<Node>> {
        match self {
            Unit::Ship(ship) => ship.navigate(root, destinations),
            Unit::Fleet(fleet) => fleet.navigate(root, destinations),
        }
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        match self {
            Unit::Ship(ship) => ship.traverse(root, destination),
            Unit::Fleet(fleet) => fleet.traverse(root, destination),
        }
    }
    fn maneuver(&self, root: &Root) -> Option<Arc<Node>> {
        match self {
            Unit::Ship(ship) => ship.maneuver(root),
            Unit::Fleet(fleet) => fleet.maneuver(root),
        }
    }
    fn try_defect(&self, root: &Root) -> Option<Vec<Arc<Node>>> {
        match self {
            Unit::Ship(ship) => ship.try_defect(root),
            Unit::Fleet(fleet) => fleet.try_defect(root),
        }
    }
    fn kill(&self) {
        match self {
            Unit::Ship(ship) => ship.kill(),
            Unit::Fleet(fleet) => fleet.kill(),
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

#[derive(Debug, Clone)]
pub enum ObjectiveFlavor {
    ReachNode {
        scalars: ObjectiveScalars,
        node: Arc<Node>,
    },
    ShipDeath {
        scalars: ObjectiveScalars,
        ship: Arc<ShipInstance>,
    },
    ShipSafe {
        scalars: ObjectiveScalars,
        ship: Arc<ShipInstance>,
        nturns: u64,
    },
    FleetDeath {
        scalars: ObjectiveScalars,
        fleet: Arc<FleetInstance>,
    },
    FleetSafe {
        scalars: ObjectiveScalars,
        fleet: Arc<FleetInstance>,
        nturns: u64,
        strengthfraction: f32,
    },
    NodeCapture {
        scalars: ObjectiveScalars,
        node: Arc<Node>,
    },
    NodeSafe {
        scalars: ObjectiveScalars,
        node: Arc<Node>,
        nturns: u64,
    },
    SystemCapture {
        scalars: ObjectiveScalars,
        system: Arc<System>,
    },
    SystemSafe {
        scalars: ObjectiveScalars,
        system: Arc<System>,
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
    fleet: Arc<FleetInstance>,
    objectives: Vec<Objective>,
}

#[derive(Debug, Clone)]
pub struct EngagementPrep {
    turn: u64,
    attackers: HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)>,
    defenders: HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)>,
    attacker_reinforcements:
        HashMap<Arc<Faction>, Vec<(u64, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>))>>,
    defender_reinforcements:
        HashMap<Arc<Faction>, Vec<(u64, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>))>>,
    location: Arc<Node>,
    aggressor: Arc<Faction>,
}

impl EngagementPrep {
    pub fn engagement_prep(root: &Root, location: Arc<Node>, aggressor: Arc<Faction>) -> Self {
        let belligerents = location.clone().get_node_forces(root);

        //at present there can be only one attacker faction
        //we figure out which of the belligerents is the aggressor
        //then get the attacker's fleets and ships
        let attackers: HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)> =
            belligerents
                .iter()
                .filter(|(faction, _)| *faction.clone() == aggressor)
                .map(|(faction, (fs, ss))| {
                    (
                        faction.clone(),
                        (
                            fs.clone(),
                            ss.iter().cloned().collect::<Vec<Arc<ShipInstance>>>(),
                        ),
                    )
                })
                .collect();

        //we do the same for defenders, the only difference being that there can be multiple defender factions
        //we check whether each faction whose assets are currently occupying the location node is at war with the attacker
        let defenders: HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)> =
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
                            ss.iter().cloned().collect::<Vec<Arc<ShipInstance>>>(),
                        ),
                    )
                })
                .collect();

        //we go through the location's neighbors and find any ships/fleets allied with the attackers' faction
        //then for each node, we get the scaling factor for travel time -- what percentage of the battle's duration the unit will be present for
        //here we don't strip out ships that are in fleets; we do that later in let-attackers
        let attacker_reinforcements: HashMap<
            Arc<Faction>,
            Vec<(u64, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>))>,
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
                                n.clone().get_distance(location.clone()),
                                n.get_node_faction_reinforcements(
                                    location.clone(),
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
            Vec<(u64, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>))>,
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
                                n.clone().get_distance(location.clone()),
                                n.get_node_faction_reinforcements(
                                    location.clone(),
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
            turn: root.turn.load(atomic::Ordering::Relaxed),
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
    attackers: HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)>,
    defenders: HashMap<Arc<Faction>, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>)>,
    aggressor: Arc<Faction>,
    objectives: HashMap<Arc<Faction>, Vec<ObjectiveFlavor>>,
    location: Arc<Node>,
    duration: u64,
    victor: Vec<Arc<Faction>>,
    ship_status: HashMap<Arc<ShipInstance>, (u64, Vec<u64>, UnitLocation)>,
    fleet_status: HashMap<Arc<FleetInstance>, Arc<Node>>,
}

impl Engagement {
    pub fn battle_cleanup(&self, root: &Root) {
        if self.victor.contains(&self.aggressor.clone()) {
            self.location.mutables.write().unwrap().allegiance = self.aggressor.clone()
        };
        for (fleet, l) in &self.fleet_status {
            fleet.transfer(UnitLocation::Node(l.clone()));
        }
        for (ship, (damage, engine_damage, location)) in &self.ship_status {
            ship.transfer(location.clone());
            {
                let mut mutables = ship.mutables.write().unwrap();
                mutables.hull = mutables.hull.saturating_sub(*damage);
                engine_damage
                    .iter()
                    .zip(
                        mutables
                            .engines
                            .iter_mut()
                            .filter(|engine| engine.health.is_some()),
                    )
                    .for_each(|(d, e)| {
                        e.health = Some(e.health.unwrap().saturating_sub(*d));
                    });
            }
            let hull = ship.mutables.read().unwrap().hull;
            if hull > 0 {
                ship.repair(true)
            };
        }
        root.remove_dead();
        root.disband_fleets();
        root.engagements
            .write()
            .unwrap()
            .push(Arc::new(self.clone()));
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
    fn get_value(self, node: Arc<Node>, faction: Arc<Faction>, battle_duration: u64)
        -> Option<f32>;
}

//this method retrieves threat value generated by a given faction
impl Salience<polarity::Supply> for Arc<Faction> {
    const DEG_MULT: f32 = 0.5;
    fn get_value(
        self,
        node: Arc<Node>,
        faction: Arc<Faction>,
        battle_duration: u64,
    ) -> Option<f32> {
        let node_strength: u64 = node.get_strength(self.clone(), battle_duration);
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
        _node: Arc<Node>,
        _faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        None
    }
}

//this method tells us how much supply there is of a given resource in a given node
impl Salience<polarity::Supply> for Arc<Resource> {
    const DEG_MULT: f32 = 1.0;
    fn get_value(
        self,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        //NOTE: Currently this does not take input stockpiles of any kind into account. We may wish to change this.
        //we add up all the resource quantity in factory output stockpiles in the node
        let factorysupply: u64 = if node.mutables.read().unwrap().allegiance == faction {
            node.mutables
                .read()
                .unwrap()
                .factoryinstancelist
                .iter()
                .map(|factory| factory.get_resource_supply_total(self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //then all the valid resource quantity in ships
        let shipsupply: u64 = node
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.is_ship())
            .filter(|ship| ship.get_allegiance() == faction)
            .map(|ship| ship.get_resource_supply(self.clone()))
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
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        //add up resources from factory input stockpiles in node
        let factorydemand: u64 = if node.mutables.read().unwrap().allegiance == faction {
            node.mutables
                .read()
                .unwrap()
                .factoryinstancelist
                .iter()
                .map(|factory| factory.get_resource_demand_total(self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //add up resources from shipyard input stockpiles in node
        let shipyarddemand: u64 = if node.mutables.read().unwrap().allegiance == faction {
            node.mutables
                .read()
                .unwrap()
                .shipyardinstancelist
                .iter()
                .map(|shipyard| shipyard.get_resource_demand_total(self.clone()))
                .sum::<u64>()
        } else {
            0
        };
        //now we have to look at ships in the node, since they might have stockpiles of their own
        let shipdemand: u64 = node
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.is_ship())
            .filter(|ship| ship.get_allegiance() == faction)
            .map(|ship| ship.get_resource_demand(self.clone()))
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
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        let sum = node
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.is_ship())
            .filter(|ship| ship.get_allegiance() == faction)
            .map(|ship| ship.get_shipclass_supply(self.clone()))
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
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        let sum = node
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.is_ship())
            .filter(|ship| ship.get_allegiance() == faction)
            .map(|ship| ship.get_shipclass_demand(self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum)
        }
    }
}

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}

//TODO: make the logic apply more generally to stockpiles attached to ships

#[derive(Debug)]
pub struct GlobalSalience {
    pub factionsalience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub resourcesalience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub shipclasssalience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
}

#[derive(Debug)]
pub struct Root {
    pub config: Config,
    pub nodeflavors: Vec<Arc<NodeFlavor>>,
    pub nodes: Vec<Arc<Node>>,
    pub systems: Vec<Arc<System>>,
    pub edgeflavors: Vec<Arc<EdgeFlavor>>,
    pub edges: HashMap<(Arc<Node>, Arc<Node>), Arc<EdgeFlavor>>,
    pub neighbors: HashMap<Arc<Node>, Vec<Arc<Node>>>,
    pub factions: Vec<Arc<Faction>>,
    pub wars: HashSet<(Arc<Faction>, Arc<Faction>)>,
    pub resources: Vec<Arc<Resource>>,
    pub hangarclasses: Vec<Arc<HangarClass>>,
    pub hangarinstancecounter: Arc<AtomicU64>,
    pub engineclasses: Vec<Arc<EngineClass>>,
    pub repairerclasses: Vec<Arc<RepairerClass>>,
    pub factoryclasses: Vec<Arc<FactoryClass>>,
    pub shipyardclasses: Vec<Arc<ShipyardClass>>,
    pub shipais: Vec<Arc<ShipAI>>,
    pub shipclasses: Vec<Arc<ShipClass>>,
    pub fleetclasses: Vec<Arc<FleetClass>>,
    pub shipinstances: RwLock<Vec<Arc<ShipInstance>>>,
    pub fleetinstances: RwLock<Vec<Arc<FleetInstance>>>,
    pub unitcounter: Arc<AtomicU64>,
    pub engagements: RwLock<Vec<Arc<Engagement>>>,
    pub globalsalience: GlobalSalience,
    pub turn: Arc<AtomicU64>,
}

impl Root {
    /*pub fn balance_stockpiles(
        &mut self,
        nodeid: Arc<Node>,
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
        let ships_in_node: Vec<(Arc<ShipInstance>, ShipInstance)> = self
            .shipinstances
            .iter()
            .filter(|(id, ship)| ship.get_mother_node(&self)
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
        &self,
        _nodeid: Arc<Node>,
        _faction: Arc<Faction>,
        _salience_map: Vec<f32>,
    ) {
    }
    //this is the method for creating a ship
    //duh
    pub fn create_ship(
        &self,
        class: Arc<ShipClass>,
        location: UnitLocation,
        faction: Arc<Faction>,
    ) -> Arc<ShipInstance> {
        //we call the shipclass instantiate method, and feed it the parameters it wants
        //let index_lock = RwLock::new(self.shipinstances);
        let new_ship = Arc::new(ShipClass::instantiate(
            class.clone(),
            location.clone(),
            faction,
            self,
        ));
        class.build_hangars(
            new_ship.clone(),
            &self.shipclasses,
            &self.hangarinstancecounter,
        );
        //NOTE: Is this thread-safe? There might be enough space in here
        //for something to go interact with the shipinstance in root and fail to get the arc from location.
        self.shipinstances.write().unwrap().push(new_ship.clone());
        location.insert_unit(new_ship.new_unit());
        new_ship
    }
    pub fn engagement_check(&self, node: Arc<Node>, actor: Arc<Faction>) -> Option<Arc<Faction>> {
        let factions = node.get_node_factions(&self);
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
                    .filter(|ship| ship.is_in_node())
                    .map(|ship| ship.get_strength(self.config.battlescalars.avg_duration))
                    .sum::<u64>()
            })
            .sum::<u64>()
            + data
                .attackers
                .iter()
                .map(|(_, (fs, _))| {
                    fs.iter()
                        .map(|fleet| fleet.get_strength(self.config.battlescalars.avg_duration))
                        .sum::<u64>()
                })
                .sum::<u64>()) as i64;
        let defender_rough_strength: i64 = (data
            .defenders
            .iter()
            .map(|(_, (_, ss))| {
                ss.iter()
                    .filter(|ship| ship.is_in_node())
                    .map(|ship| ship.get_strength(self.config.battlescalars.avg_duration))
                    .sum::<u64>()
            })
            .sum::<u64>()
            + data
                .defenders
                .iter()
                .map(|(_, (fs, _))| {
                    fs.iter()
                        .map(|fleet| fleet.get_strength(self.config.battlescalars.avg_duration))
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
                    .map(|fleet| fleet.mutables.read().unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|ship| !ship.is_in_fleet())
                        .map(|ship| ship.mutables.read().unwrap().objectives.clone())
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
                        .map(|fleet| fleet.mutables.read().unwrap().objectives.clone())
                        .flatten()
                        .collect::<Vec<ObjectiveFlavor>>();
                    d.append(
                        &mut ss
                            .iter()
                            .filter(|ship| !ship.is_in_fleet())
                            .map(|ship| ship.mutables.read().unwrap().objectives.clone())
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
            Vec<(f32, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>))>,
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
            Vec<(f32, (Vec<Arc<FleetInstance>>, Vec<Arc<ShipInstance>>))>,
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
                    .filter(|ship| !ship.is_in_fleet())
                    .map(|ship| ship.get_strength(duration))
                    .sum::<u64>()
                    + fs.iter()
                        .map(|fleet| fleet.get_strength(duration))
                        .sum::<u64>()
            })
            .sum::<u64>()
            + scaled_attacker_reinforcements
                .iter()
                .map(|(_, v)| {
                    v.iter()
                        .map(|(scalar, (fs, ss))| {
                            ((ss.iter()
                                .filter(|ship| !ship.is_in_fleet())
                                .map(|ship| ship.get_strength(duration))
                                .sum::<u64>()
                                + fs.iter()
                                    .map(|fleet| fleet.get_strength(duration))
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
                    .filter(|ship| !ship.is_in_fleet())
                    .map(|ship| ship.get_strength(duration))
                    .sum::<u64>()
                    + fs.iter()
                        .map(|fleet| fleet.get_strength(duration))
                        .sum::<u64>()
            })
            .sum::<u64>()
            + scaled_defender_reinforcements
                .iter()
                .map(|(_fid, v)| {
                    v.iter()
                        .map(|(scalar, (fs, ss))| {
                            ((ss.iter()
                                .filter(|ship| !ship.is_in_fleet())
                                .map(|ship| ship.get_strength(duration))
                                .sum::<u64>()
                                + fs.iter()
                                    .map(|fleet| fleet.get_strength(duration))
                                    .sum::<u64>()) as f32
                                * scalar) as u64
                        })
                        .sum::<u64>()
                })
                .sum::<u64>();

        //we don't take the objectives of reinforcement units into account
        let attacker_objective_difficulty: f32 = data
            .attackers
            .iter()
            .map(|(_, (fs, ss))| {
                let mut d = fs
                    .iter()
                    .map(|fleet| fleet.mutables.read().unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|ship| !ship.is_in_fleet())
                        .map(|ship| ship.mutables.read().unwrap().objectives.clone())
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
                    .map(|fleet| fleet.mutables.read().unwrap().objectives.clone())
                    .flatten()
                    .collect::<Vec<ObjectiveFlavor>>();
                d.append(
                    &mut ss
                        .iter()
                        .filter(|ship| !ship.is_in_fleet())
                        .map(|ship| ship.mutables.read().unwrap().objectives.clone())
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

        let all_fleets: Vec<Arc<FleetInstance>> = data
            .attackers
            .iter()
            .map(|(_fid, (fs, _ss))| fs.clone())
            .chain(data.defenders.iter().map(|(_fid, (fs, _ss))| fs.clone())) //NOTE: Can we avoid this clone?
            .chain(
                scaled_attacker_reinforcements
                    .iter()
                    .map(|(_fid, vec)| vec.iter().map(|(_fid, (fs, _ss))| fs.clone()))
                    .flatten(),
            )
            .chain(
                scaled_defender_reinforcements
                    .iter()
                    .map(|(_fid, vec)| vec.iter().map(|(_fid, (fs, _ss))| fs.clone()))
                    .flatten(),
            )
            .flatten()
            .collect();

        let all_ships: Vec<Arc<ShipInstance>> = data
            .attackers
            .iter()
            .map(|(_fid, (_fs, ss))| ss.clone())
            .chain(data.defenders.iter().map(|(_fid, (_fs, ss))| ss.clone())) //NOTE: Can we avoid this clone?
            .chain(
                scaled_attacker_reinforcements
                    .iter()
                    .map(|(_fid, vec)| vec.iter().map(|(_fid, (_fs, ss))| ss.clone()))
                    .flatten(),
            )
            .chain(
                scaled_defender_reinforcements
                    .iter()
                    .map(|(_fid, vec)| vec.iter().map(|(_fid, (_fs, ss))| ss.clone()))
                    .flatten(),
            )
            .flatten()
            .collect();

        let neighbors = self.neighbors.get(&data.location).unwrap();

        let fleet_status: HashMap<Arc<FleetInstance>, Arc<Node>> = all_fleets
            .iter()
            .map(|fleet| {
                (
                    fleet.clone(),
                    fleet
                        .navigate(self, neighbors)
                        .unwrap_or(data.location.clone()),
                )
            })
            .collect();

        let duration_damage_rand = Normal::<f32>::new(1.0, self.config.battlescalars.damage_dev)
            .unwrap()
            .sample(&mut rand::thread_rng())
            .clamp(0.0, 1.0);

        //NOTE: Maybe have the lethality scaling over battle duration be logarithmic? Maybe modder-specified?
        let ship_status: HashMap<Arc<ShipInstance>, (u64, Vec<u64>, UnitLocation)> = {
            all_ships
                .iter()
                .map(|ship| {
                    let rand_factor =
                        Normal::<f32>::new(0.25, self.config.battlescalars.damage_dev)
                            .unwrap()
                            .sample(&mut rand::thread_rng())
                            .clamp(0.0, 10.0);
                    if !victor.contains(&ship.get_allegiance()) {
                        let new_location = if ship.is_in_node() {
                            UnitLocation::Node(
                                ship.navigate(self, neighbors)
                                    .unwrap_or(data.location.clone()),
                            )
                        } else {
                            ship.mutables.read().unwrap().location.clone()
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
                            .mutables
                            .read()
                            .unwrap()
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
                        (ship.clone(), (damage, engine_damage, new_location))
                    } else {
                        let new_location = if ship
                            .get_mother_fleet()
                            .map(|fleet| all_fleets.contains(&fleet))
                            .unwrap_or(false)
                        {
                            ship.mutables.read().unwrap().location.clone()
                        } else {
                            UnitLocation::Node(data.location.clone())
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
                            .mutables
                            .read()
                            .unwrap()
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
                        (ship.clone(), (damage, engine_damage, new_location))
                    }
                })
                .collect()
        };

        Engagement {
            visiblename: format!("Battle of {}", data.location.visiblename.clone()),
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
    pub fn remove_dead(&self) {
        let dead: Vec<Arc<ShipInstance>> = self
            .shipinstances
            .read()
            .unwrap()
            .iter()
            .filter(|ship| ship.mutables.read().unwrap().hull == 0)
            .cloned()
            .collect();
        dead.iter().for_each(|ship| {
            ship.clone().kill();
        });
        self.shipinstances
            .write()
            .unwrap()
            .retain(|ship| ship.mutables.read().unwrap().hull > 0);
    }
    pub fn disband_fleets(&self) {
        let dead = self
            .fleetinstances
            .read()
            .unwrap()
            .iter()
            .filter(|fleet| {
                ((fleet.get_strength(self.config.battlescalars.avg_duration) as f32)
                    < (fleet.idealstrength as f32 * fleet.class.disbandthreshold))
                    && !fleet.mutables.read().unwrap().phantom
            })
            .cloned()
            .collect::<Vec<_>>();
        for fleet in dead {
            fleet
                .get_daughters()
                .iter()
                .all(|daughter| daughter.transfer(UnitLocation::Node(fleet.get_mother_node())));
        }
        let remaining: Vec<Arc<FleetInstance>> = self
            .fleetinstances
            .read()
            .unwrap()
            .iter()
            .filter(|fleet| {
                fleet.mutables.read().unwrap().phantom || !fleet.get_daughters().is_empty()
            })
            .cloned()
            .collect();
        self.fleetinstances
            .write()
            .unwrap()
            .retain(|fleet| remaining.contains(fleet));
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
        let node_initial_salience_map: Vec<(Arc<Node>, f32)> = self
            .nodes
            .iter()
            .filter_map(|node| {
                salience
                    .clone()
                    .get_value(
                        node.clone(),
                        subject_faction.clone(),
                        self.config.battlescalars.avg_duration,
                    )
                    .map(|v| (node.clone(), v))
            })
            .collect();
        //this map contains the amount of threat that exists from each faction, in each node, from the perspective of the subject faction
        //Length equals all nodes
        //This is a subjective map for subject faction
        let tagged_threats: Vec<Vec<[f32; 2]>> = transpose(
            self.globalsalience.factionsalience.read().unwrap()[subject_faction.id].clone(),
        );
        //this is the factor by which a salience passing through each node should be multiplied
        //we sum the tagged threats for each node -- which are valenced according to relations with the subject faction
        //then we use Alyssa's black mathemagics to convert them so that the scaling curve is correct
        //Length equals all nodes
        //This is a subjective map for subject faction
        let node_degradations: Vec<f32> = tagged_threats
            .iter()
            .map(|factions_vec| {
                let sum = factions_vec.iter().map(|[supply, _]| supply).sum();
                scale_from_threat(sum, 20_f32) * S::DEG_MULT * 0.8
            })
            .collect();
        //Outer vec length equals all nodes; inner vec equals nodes owned by faction and producing specified salience -- but only the inner node corresponding to the outer node has a nonzero value
        let node_salience_state: Vec<Vec<f32>> = self
            .nodes
            .iter()
            .map(|node| {
                //we iterate over the node initial salience map, which contains only nodes owned by subject faction and producing subject salience
                node_initial_salience_map
                    .iter()
                    //that gives us the initial salience value for each node
                    //we use this '== check as u8' to multiply it by 1 if the node matches the one the outer iterator is looking at, and multiply it by 0 otherwise
                    .map(|(sourcenode, value)| {
                        value * ((sourcenode.clone() == node.clone()) as u8) as f32
                    })
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
                    let deg_a = node_degradations[a.id];
                    let deg_b = node_degradations[b.id];
                    //this loop does basically the same thing as an iterator but we have to do it this way for complicated ownership reasons
                    //we repeat the loop process n_tags times, 
                    for i in 0..n_tags {
                        //we index into node_salience_state's outer vec by node A's id, then into the inner vec by i; this means we're essentially iterating over the inner vec
                        //we update the i'th element of A (the inner vec) by taking the maximum between the i'th element of A and the i'th element of B, multiplied by node B's degradation scalar
                        //because this is the salience coming from node B to node A, getting degraded by B's threats as it leaves
                        state[a.id][i] = state[a.id][i].max(state[b.id][i] * deg_b);
                        //then we do the same thing again but backwards, to process the salience coming from node A to node B
                        state[b.id][i] = state[b.id][i].max(state[a.id][i] * deg_a);
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
    pub fn calculate_global_faction_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .par_iter()
            .map(|subjectfaction| {
                self.factions
                    .par_iter()
                    .map(|objectfaction| {
                        let supply = self.calculate_values::<Arc<Faction>, polarity::Supply>(
                            objectfaction.clone(),
                            subjectfaction.clone(),
                            self.config.saliencescalars.faction_prop_iters,
                        );
                        let demand = self.calculate_values::<Arc<Faction>, polarity::Demand>(
                            objectfaction.clone(),
                            subjectfaction.clone(),
                            self.config.saliencescalars.faction_prop_iters,
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
    pub fn calculate_global_resource_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .par_iter()
            .map(|faction| {
                self.resources
                    .par_iter()
                    .map(|resource| {
                        let supply = self.calculate_values::<Arc<Resource>, polarity::Supply>(
                            resource.clone(),
                            faction.clone(),
                            self.config.saliencescalars.resource_prop_iters,
                        );
                        let demand = self.calculate_values::<Arc<Resource>, polarity::Demand>(
                            resource.clone(),
                            faction.clone(),
                            self.config.saliencescalars.resource_prop_iters,
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
            .par_iter()
            .map(|faction| {
                self.shipclasses
                    .par_iter()
                    .map(|shipclass| {
                        let supply = self.calculate_values::<Arc<ShipClass>, polarity::Supply>(
                            shipclass.clone(),
                            faction.clone(),
                            self.config.saliencescalars.shipclass_prop_iters,
                        );
                        let demand = self.calculate_values::<Arc<ShipClass>, polarity::Demand>(
                            shipclass.clone(),
                            faction.clone(),
                            self.config.saliencescalars.shipclass_prop_iters,
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
        let start0 = Instant::now();
        //increment turn counter
        let turn = self.turn.fetch_add(1, atomic::Ordering::Relaxed);
        println!("It is now turn {}.", turn);

        //reset all ships' engines
        self.shipinstances
            .write()
            .unwrap()
            .iter()
            .for_each(|ship| ship.reset_movement());

        //run all ship repairers
        self.shipinstances
            .write()
            .unwrap()
            .iter()
            .for_each(|ship| ship.repair(false));

        //process all factories
        self.nodes.iter().for_each(|n| n.process_factories());
        self.shipinstances
            .write()
            .unwrap()
            .iter()
            .for_each(|ship| ship.process_factories());

        //process all shipyards
        self.nodes.iter().for_each(|n| n.process_shipyards());
        self.shipinstances
            .write()
            .unwrap()
            .iter()
            .for_each(|ship| ship.process_shipyards());

        //plan ship creation
        let ship_plan_list: Vec<(Arc<ShipClass>, UnitLocation, Arc<Faction>)> = self
            .nodes
            .iter()
            .map(|node| node.clone().plan_ships(&self.shipclasses))
            .chain(
                self.shipinstances
                    .write()
                    .unwrap()
                    .iter()
                    .map(|ship| ship.plan_ships(&self.shipclasses)),
            )
            .flatten()
            .collect();
        //create queued ships
        let n_newships = ship_plan_list
            .iter()
            .map(|(id, location, faction)| {
                self.create_ship(id.clone(), location.clone(), faction.clone())
            })
            .count();
        println!("Built {} new ships.", n_newships);

        //propagate threat values
        //propagate saliences, create salience map
        //NOTE: I'm thinking of setting this up such that we use last turn's threat salience map as the degradation map for this turn's threat salience propagation.
        //That's what'll happen here with the way it is currently (barring the update_threats loop above). Previous turn's values get used in calc faction salience, then get overwritten.
        //At least assuming the order of operations works like I'm expecting -- check that with Amaryllis.
        //On turn 1, we'll want to run calc faction salience several times to get the values to settle.
        let start1 = Instant::now();
        let gfs = self.calculate_global_faction_salience();
        *self.globalsalience.factionsalience.write().unwrap() = gfs;
        let grs = self.calculate_global_resource_salience();
        *self.globalsalience.resourcesalience.write().unwrap() = grs;
        let gus = self.calculate_global_shipclass_salience();
        *self.globalsalience.shipclasssalience.write().unwrap() = gus;
        let duration1 = start1.elapsed();
        dbg!(duration1);

        //run operation management logic

        //move ships, one edge at a time
        //running battle checks and stockpile balancing with each traversal
        let start = Instant::now();
        let shipinstances = self.shipinstances.read().unwrap().clone();
        shipinstances.iter().for_each(|shipinstance| {
            if let Some(destination) = shipinstance.maneuver(self) {
                if let Some(aggressor) =
                    self.engagement_check(destination.clone(), shipinstance.get_allegiance())
                {
                    let engagement = self.internal_battle(EngagementPrep::engagement_prep(
                        self,
                        destination,
                        aggressor,
                    ));
                    engagement.battle_cleanup(&self);
                }
            }
        });
        dbg!(start.elapsed());

        //move fleets, one edge at a time
        //running battle checks and stockpile balancing with each traversal

        //run defection logic

        //run diplomacy logic

        //transmit root data to frontend

        //NOTE: I don't remember what this is for
        //I don't *think* it's doing anything that actually matters, but Chesterton's Fence?
        //self.nodes.iter().for_each(|node| {
        //    let mut threat_list: Vec<(Arc<Faction>, f32)> = node
        //        .mutables
        //        .read()
        //        .unwrap()
        //        .threat
        //        .iter()
        //        .map(|(faction, v)| (faction.clone(), *v))
        //        .collect();
        //    threat_list.sort_by_key(|(faction, _)| faction.clone());
        //});
        dbg!(start0.elapsed());
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
