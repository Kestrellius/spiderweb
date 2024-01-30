use crate::connection;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_distr::*;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
use std::sync::{RwLock, RwLockWriteGuard};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub saliencescalars: SalienceScalars,
    pub entityscalars: EntityScalars,
    pub battlescalars: BattleScalars,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SalienceScalars {
    pub faction_deg_mult: f32,
    pub resource_deg_mult: f32,
    pub unitclass_deg_mult: f32,
    pub faction_prop_iters: usize, //number of edges across which this salience will propagate during a turn
    pub resource_prop_iters: usize,
    pub unitclass_prop_iters: usize,
    pub volume_strength_ratio: f32, //multiplier for resource/unitclass supply points when comparing to threat values for faction demand calcs
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityScalars {
    pub defect_escape_scalar: f32,
    pub victor_morale_scalar: f32,
    pub victis_morale_scalar: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct NodeMut {
    pub visibility: bool,
    pub flavor: Arc<NodeFlavor>, //type of location this node is -- planet, asteroid field, hyperspace transit zone
    pub units: Vec<Unit>,
    pub factoryinstancelist: Vec<FactoryInstance>, //this is populated at runtime from the factoryclasslist, not specified in json
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub allegiance: Arc<Faction>, //faction that currently holds the node
    pub efficiency: f32, //efficiency of any production facilities in this node; changes over time based on faction ownership
    pub balance_stockpiles: bool,
    pub balance_hangars: bool,
    pub check_for_battles: bool,
    pub stockpiles_balanced: bool,
    pub hangars_balanced: bool,
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
    pub fn get_node_forces(&self, root: &Root) -> HashMap<Arc<Faction>, Vec<Unit>> {
        root.factions
            .iter()
            .map(|faction| {
                let units: Vec<Unit> = self
                    .mutables
                    .read()
                    .unwrap()
                    .units
                    .iter()
                    .filter(|unit| unit.get_allegiance() == *faction)
                    .cloned()
                    .collect();
                (faction.clone(), units)
            })
            .filter(|(_, units)| units.len() > 0)
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
    ) -> Vec<Unit> {
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
        top_level_units
            .into_iter()
            .chain(daughter_units.into_iter())
            .collect()
    }
    pub fn get_node_faction_forces(
        &self,
        faction: Arc<Faction>,
    ) -> (Vec<Arc<SquadronInstance>>, Vec<Arc<ShipInstance>>) {
        let ships: Vec<Arc<ShipInstance>> = self
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter_map(|unit| unit.get_ship())
            .filter(|ship| ship.mutables.read().unwrap().allegiance == faction)
            .collect();
        let squadrons: Vec<Arc<SquadronInstance>> = self
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter_map(|unit| unit.get_squadron())
            .filter(|squadron| squadron.mutables.read().unwrap().allegiance == faction)
            .collect();
        (squadrons, ships)
    }
    pub fn get_unitclass_num(&self, shipclass: Arc<ShipClass>) -> u64 {
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

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.visiblename == other.visiblename
            && self.position == other.position
            && self.description == other.description
            && self.environment == other.environment
            && self.bitmap == other.bitmap
            && self.mutables.read().unwrap().clone() == other.mutables.read().unwrap().clone()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeFlavor {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub propagates: bool,
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

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct FactionID {
    pub index: usize,
}

impl FactionID {
    pub fn new_from_index(index: usize) -> Self {
        FactionID { index: index }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Faction {
    pub id: usize,
    pub visiblename: String, //faction name as shown to player
    pub description: String,
    pub visibility: bool,
    pub propagates: bool,
    pub efficiencydefault: f32, //starting value for production facility efficiency
    pub efficiencytarget: f32, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    pub efficiencydelta: f32,  //rate at which efficiency changes
    pub battlescalar: f32,
    pub value_mult: f32, //how valuable the AI considers one point of this faction's threat to be
    pub volume_strength_ratio: f32, //faction's multiplier for resource/unitclass supply points when comparing to threat values for faction demand calcs
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub propagates: bool,
    pub unit_vol: u64, //how much volume (in e.g. cubic meters) one unit of this resource takes up; this is intended only for player-facing use
    pub value_mult: f32, //how valuable the AI considers one unit of this resource to be
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

//a unipotent resource stockpile can contain only one type of resource
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
    pub propagates: bool,
}

impl UnipotentStockpile {
    fn input_is_sufficient(&self) -> bool {
        self.contents >= self.rate
    }
    //this is the logic to determine whether a unipotent stockpile should be active, dormant, or stalled
    fn output_state(&self) -> OutputState {
        //NOTE: Dormancy is dummied out for now
        //if self.contents >= self.target {
        //    OutputState::Dormant
        //} else
        if self.contents + self.rate >= self.capacity {
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

//a pluripotent stockpile can contain any number of different resources and ships
//however, it has no constant rate of increase or decrease; things may only be added or removed manually
#[derive(Debug, Clone, PartialEq)]
pub struct PluripotentStockpile {
    pub visibility: bool,
    pub contents: HashMap<Arc<Resource>, u64>,
    pub allowed: Option<Vec<Arc<Resource>>>,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl Stockpileness for PluripotentStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        self.contents.clone()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        *self.contents.get(&cargo).unwrap_or(&0)
    }
    fn get_capacity(&self) -> u64 {
        self.capacity
    }
    fn get_fullness(&self) -> u64 {
        self.contents.iter().map(|(_, value)| value).sum::<u64>()
    }
    fn get_allowed(&self) -> Option<Vec<Arc<Resource>>> {
        self.allowed.clone()
    }
    //unlike other places, here in pluripotent stockpiles we don't take target into account when calculating supply
    //thus, items in pluripotent stockpiles always emit supply, even if the stockpile still wants more
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        self.get_resource_num(resource.clone())
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        let resource_vec = vec![resource.clone()];
        if self
            .get_allowed()
            .unwrap_or(resource_vec.clone())
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HangarClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub capacity: u64,                    //total volume the hangar can hold
    pub target: u64, //volume the hangar wants to hold; this is usually equal to capacity
    pub allowed: Vec<UnitClassID>, //which shipclasses this hangar can hold
    pub ideal: HashMap<UnitClassID, u64>, //how many of each ship type the hangar wants
    pub non_ideal_supply_scalar: f32, //multiplier used for demand generated by non-ideal ships under the target limit; should be below one
    pub launch_volume: u64,           //how much volume the hangar can launch at one time in battle
    pub launch_interval: u64,         //time between launches in battle
    pub propagates: bool,             //whether or not hangar generates saliences
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
            .map(|(unitclass, v)| {
                let ideal_strength = match unitclass {
                    UnitClassID::ShipClass(sc) => {
                        root.shipclasses[sc.index].get_ideal_strength(root)
                    }
                    UnitClassID::SquadronClass(fc) => {
                        root.squadronclasses[fc.index].get_ideal_strength(root)
                    }
                };
                ideal_strength * v
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
    pub fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .map(|daughter| daughter.get_unitclass_num(unitclass.clone()))
            .sum::<u64>()
    }
    pub fn get_unitclass_supply(&self, unitclass: UnitClass) -> u64 {
        let daughter_supply = self
            .mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .map(|daughter| daughter.get_unitclass_supply(unitclass.clone()))
            .sum::<u64>();
        let ideal_volume = self
            .class
            .ideal
            .get(&UnitClassID::new_from_unitclass(&unitclass))
            .unwrap_or(&0)
            * unitclass.get_ideal_volume();
        let non_ideal_volume = daughter_supply.saturating_sub(ideal_volume);
        let excess_volume = self.get_fullness().saturating_sub(self.class.target);
        let over_target_supply = (excess_volume).min(non_ideal_volume);
        let under_target_supply = ((non_ideal_volume.saturating_sub(over_target_supply)) as f32
            * self.class.non_ideal_supply_scalar) as u64;
        over_target_supply + under_target_supply
    }
    pub fn get_unitclass_demand(&self, unitclass: UnitClass) -> u64 {
        let daughter_volume = self
            .mutables
            .read()
            .unwrap()
            .contents
            .iter()
            .filter(|unit| &unit.get_unitclass() == &unitclass)
            .map(|unit| unit.get_volume())
            .sum::<u64>();
        let ideal_volume = self
            .class
            .ideal
            .get(&UnitClassID::new_from_unitclass(&unitclass))
            .unwrap_or(&0)
            * unitclass.get_ideal_volume();
        ideal_volume.saturating_sub(daughter_volume)
            + self
                .mutables
                .read()
                .unwrap()
                .contents
                .iter()
                .map(|daughter| daughter.get_unitclass_demand(unitclass.clone()))
                .sum::<u64>()
    }
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

pub trait ResourceProcess {
    fn get_state(&self) -> FactoryState;
    fn get_resource_supply_total(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64;
    fn get_target_total(&self) -> u64;
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

#[derive(Debug, Clone, PartialEq)]
pub struct EngineInstance {
    pub engineclass: Arc<EngineClass>,
    pub visibility: bool,
    pub basehealth: Option<u64>,
    pub health: Option<u64>,
    pub toughnessscalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
    pub last_move_turn: u64,
}

impl EngineInstance {
    fn check_engine(
        &self,
        root: &Root,
        location: Arc<Node>,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<(Vec<Arc<Node>>, u64)> {
        if (self.health != Some(0))
            && ((root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn) > self.cooldown)
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
    fn check_engine_movement_only(&self, turn: u64) -> bool {
        if (self.health != Some(0))
            && ((turn - self.last_move_turn) > self.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            true
        } else {
            false
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
            && ((root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn) > self.cooldown)
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
    fn get_moves_left(&self, movement_left: u64) -> u64 {
        (self
            .inputs
            .iter()
            .map(|sp| sp.contents / sp.rate)
            .min()
            .unwrap_or(0))
        .min(movement_left / self.speed)
    }
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
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
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

#[derive(Debug, Clone, PartialEq)]
pub struct RepairerInstance {
    pub repairerclass: Arc<RepairerClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerInstance {
    fn process(&mut self) {
        self.inputs
            .iter_mut()
            .for_each(|input| input.input_process());
    }
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
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterNodeWeaponClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub visibility: bool,
    pub basehealth: Option<u64>,
    pub toughnessscalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the weapon won't fire into nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the weapon won't fire across edges of these flavors
    pub damage: (u64, u64), //lower and upper bounds for damage done by a single shot
    pub engine_damage: (u64, u64), //lower and upper bounds for damage to engine done by a single shot
    pub accuracy: f32, //divided by target's internodeweaponevasionscalar to get hit probability as a fraction of 1.0
    pub range: u64,    //how many edges away the weapon can reach
    pub shots: (u64, u64), //lower and upper bounds for maximum number of shots the weapon fires each time it's activated
    pub target_priorities_class: HashMap<Arc<ShipClassID>, f32>, //how strongly weapon will prioritize ships of each class; classes absent from list will default to 1.0
    pub target_priorities_flavor: HashMap<Arc<ShipFlavor>, f32>, //how strongly weapon will prioritize ships of each flavor; flavors absent from list will default to 1.0
}

impl InterNodeWeaponClass {
    pub fn instantiate(class: Arc<Self>) -> InterNodeWeaponInstance {
        InterNodeWeaponInstance {
            class: class.clone(),
            visibility: class.visibility,
            health: class.basehealth,
            inputs: class.inputs.clone(),
        }
    }
}

impl Eq for InterNodeWeaponClass {}

impl Ord for InterNodeWeaponClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for InterNodeWeaponClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for InterNodeWeaponClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterNodeWeaponInstance {
    pub class: Arc<InterNodeWeaponClass>,
    pub visibility: bool,
    pub health: Option<u64>,
    pub inputs: Vec<UnipotentStockpile>,
}

impl InterNodeWeaponInstance {
    fn fire<R: Rng>(
        &self,
        root: &Root,
        mother: Arc<ShipInstance>,
        rng: &mut R,
    ) -> HashMap<Unit, UnitStatus> {
        let allegiance = mother.get_allegiance();
        let location = mother.get_mother_node();
        let mut target_nodes = vec![location.clone()];
        let mut node_layer = vec![location.clone()];
        for _ in 0..self.class.range {
            let neighbors: Vec<Arc<Node>> = node_layer
                .iter()
                .map(|node| {
                    root.neighbors
                        .get(node)
                        .unwrap()
                        .iter()
                        .filter(move |rhs_node| {
                            !(target_nodes.contains(&rhs_node)
                                || self
                                    .class
                                    .forbidden_nodeflavors
                                    .contains(&rhs_node.mutables.read().unwrap().flavor)
                                || self.class.forbidden_edgeflavors.contains(
                                    root.edges
                                        .get(&(
                                            node.min(rhs_node).clone(),
                                            rhs_node.max(&node).clone().clone(),
                                        ))
                                        .unwrap(),
                                ))
                        })
                })
                .flatten()
                .cloned()
                .collect();
            target_nodes.append(&mut neighbors.clone());
            node_layer = neighbors.clone();
        }

        let enemies = root
            .factions
            .iter()
            .filter(|faction| {
                root.wars.contains(&(
                    faction.clone().min(&allegiance.clone().clone()).clone(),
                    allegiance.clone().max(faction.clone().clone()).clone(),
                ))
            })
            .collect::<Vec<_>>();

        let targets = target_nodes
            .iter()
            .map(|node| {
                node.mutables
                    .read()
                    .unwrap()
                    .units
                    .iter()
                    .filter_map(|unit| unit.get_ship())
                    .filter(|ship| enemies.contains(&&ship.mutables.read().unwrap().allegiance))
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        let shots_fired: usize = rng.gen_range(self.class.shots.0..self.class.shots.1) as usize;

        let hit_ships = targets
            .iter()
            .sorted_by_key(|target| {
                NotNan::new(
                    self.class
                        .target_priorities_class
                        .get(&ShipClassID::new_from_index(target.class.id))
                        .unwrap_or(&0.0)
                        + self
                            .class
                            .target_priorities_flavor
                            .get(&target.class.shipflavor)
                            .unwrap_or(&0.0),
                )
                .unwrap()
            })
            .take(shots_fired)
            .filter(|target| {
                (self.class.accuracy / target.class.internodeweaponevasionscalar) > 1.0
            })
            .collect();

        hit_ships.iter().map(|hit_ship| {
            let status = UnitStatus {
                location: hit_ship.mutables.read().unwrap().location,
                damage: rng.gen_range(self.class.damage.0..self.class.damage.1)
                    / hit_ship.toughnessscalar,
                engine_damage: hit_ship
                    .mutables
                    .read()
                    .unwrap()
                    .engines
                    .iter()
                    .filter(|e| e.health.is_some())
                    .map(|e| {
                        (rng.gen_range(self.class.engine_damage.0..self.class.engine_damage.1)
                            / e.toughnessscalar) as u64
                    })
                    .collect(),
            };
            (hit_ship, status)
        })
    }
}

impl ResourceProcess for InterNodeWeaponInstance {
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
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
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

#[derive(Debug, Clone, PartialEq)]
pub struct FactoryInstance {
    //this is an actual factory, derived from a factory class
    pub factoryclass: Arc<FactoryClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
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
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>()
    }
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64 {
        self.inputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum::<u64>()
            + self.outputs.iter().map(|sp| sp.target).sum::<u64>()
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

#[derive(Debug, Clone, PartialEq)]
pub struct ShipyardInstance {
    pub shipyardclass: Arc<ShipyardClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<Arc<ShipClass>, u64>,
    pub constructpoints: u64,
    pub constructrate: u64,
    pub efficiency: f32,
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
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum()
    }
    fn get_target_total(&self) -> u64 {
        self.inputs.iter().map(|sp| sp.target).sum()
    }
}

#[derive(Debug, Clone)]
pub struct ShipAI {
    pub id: usize,
    pub ship_attract_specific: f32, //a multiplier for demand gradients corresponding to the specific class of a ship using this AI
    pub ship_attract_generic: f32, //a multiplier for the extent to which a ship using this AI will follow generic ship demand gradients
    pub ship_cargo_attract: HashMap<UnitClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>, //a list of resources whose demand gradients this AI will follow, and individual strength multipliers
    pub friendly_supply_attract: f32,
    pub hostile_supply_attract: f32,
    pub allegiance_demand_attract: f32,
    pub enemy_demand_attract: f32,
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

#[derive(Debug, Clone)]
pub enum UnitLocation {
    Node(Arc<Node>),
    Squadron(Arc<SquadronInstance>),
    Hangar(Arc<HangarInstance>),
}

impl UnitLocation {
    fn check_insert(&self, unit: Unit) -> bool {
        match self {
            UnitLocation::Node(_node) => true,
            UnitLocation::Squadron(squadron) => squadron
                .mutables
                .read()
                .unwrap()
                .location
                .check_insert(unit.clone()),
            UnitLocation::Hangar(hangar) => {
                unit.get_volume() <= hangar.class.capacity - hangar.get_fullness()
            }
        }
    }
    fn check_remove(&self, unit: Unit) -> bool {
        match self {
            UnitLocation::Node(node) => node.mutables.read().unwrap().units.contains(&unit),
            UnitLocation::Squadron(squadron) => squadron.get_daughters().contains(&unit),
            UnitLocation::Hangar(hangar) => {
                hangar.mutables.read().unwrap().contents.contains(&unit)
            }
        }
    }
    fn insert_unit(&self, unit: Unit) {
        match self {
            UnitLocation::Node(node) => node.mutables.write().unwrap().units.push(unit.clone()),
            UnitLocation::Squadron(squadron) => squadron
                .mutables
                .write()
                .unwrap()
                .daughters
                .push(unit.clone()),
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
            UnitLocation::Squadron(squadron) => squadron
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

impl PartialEq for UnitLocation {
    fn eq(&self, other: &Self) -> bool {
        let self_val = match self {
            UnitLocation::Node(n) => n.id,
            UnitLocation::Squadron(s) => s.id as usize,
            UnitLocation::Hangar(h) => h.id as usize,
        };
        let other_val = match other {
            UnitLocation::Node(n) => n.id,
            UnitLocation::Squadron(s) => s.id as usize,
            UnitLocation::Hangar(h) => h.id as usize,
        };
        self_val == other_val
    }
}

pub trait Mobility {
    fn get_unit(&self) -> Unit;
    fn get_unitclass(&self) -> UnitClass;
    fn get_ship(&self) -> Option<Arc<ShipInstance>>;
    fn get_squadron(&self) -> Option<Arc<SquadronInstance>>;
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
    fn is_in_squadron(&self) -> bool {
        match self.get_location() {
            UnitLocation::Squadron(_) => true,
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
            UnitLocation::Squadron(squadron) => squadron.get_mother_node(),
            UnitLocation::Hangar(hangar) => hangar.mother.get_mother_node(),
        }
    }
    fn get_mother_unit(&self) -> Option<Unit> {
        match self.get_location() {
            UnitLocation::Node(_) => None,
            UnitLocation::Squadron(squadron) => Some(squadron.get_unit()),
            UnitLocation::Hangar(hangar) => Some(hangar.mother.get_unit()),
        }
    }
    fn get_hull(&self) -> u64;
    fn get_allegiance(&self) -> Arc<Faction>;
    fn get_daughters(&self) -> Vec<Unit>;
    fn get_daughters_recursive(&self) -> Vec<Unit>;
    fn get_morale_scalar(&self) -> f32;
    fn get_character_strength_scalar(&self) -> f32;
    fn get_interdiction_scalar(&self) -> f32;
    fn get_processordemandnavscalar(&self) -> f32;
    fn get_strength(&self, time: u64) -> u64;
    fn get_strength_post_engagement(&self, damage: u64) -> u64;
    fn get_volume(&self) -> u64;
    fn get_ai(&self) -> NavAI;
    fn get_navthreshold(&self) -> f32;
    fn get_objectives(&self) -> Vec<Objective>;
    fn get_deployment_threshold(&self) -> Option<u64>;
    fn get_deployment_status(&self) -> bool;
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32);
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64;
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64;
    fn get_unitclass_supply(&self, unitclass: UnitClass) -> u64;
    fn get_unitclass_demand(&self, unitclass: UnitClass) -> u64;
    fn change_allegiance(&self, new_faction: Arc<Faction>);
    fn acyclicity_check(&self, location: UnitLocation) -> bool {
        match location.clone() {
            UnitLocation::Squadron(squadron) => {
                if squadron.get_id() == self.get_id() {
                    false
                } else {
                    self.acyclicity_check(squadron.get_location())
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
    //The ship and squadron implementations of this next method do slightly different things.
    //The ship version is used for gathering reinforcements, and assumes the ship can't make the move and its daughters will have to move independently.
    //It recurses down the tree, following that logic at every stage.
    //The squadron version is used to determine how much of the squadron can make a particular move.
    //Since granddaughters won't leave their mothers to accompany a squadron that leaves them beihnd, this version just checks the immediate daughters.
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit>;
    fn set_movement_recursive(&self, value: u64);
    fn get_moves_left(&self, turn: u64) -> u64;
    fn process_engines(&self, root: &Root, destination: Arc<Node>);
    fn get_node_nav_attractiveness(&self, root: &Root, node: Arc<Node>) -> NotNan<f32> {
        let location: Arc<Node> = self.get_mother_node();
        let allegiance = self.get_allegiance();
        let ai = self.get_ai();

        let resource_salience = &root.globalsalience.resourcesalience.read().unwrap();

        //this checks how much value the node holds with regards to resources the subject ship is seeking
        let resource_demand_value: f32 = ai
            .resource_attract
            .iter()
            .map(|(resource, scalar)| {
                let demand = resource_salience[allegiance.id][resource.id][node.id][0];
                let supply = resource_salience[allegiance.id][resource.id][node.id][1];
                (demand - supply) * self.get_resource_supply(resource.clone()) as f32 * scalar
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
                let demand = resource_salience[allegiance.id][resource.id][node.id][0];
                let supply = resource_salience[allegiance.id][resource.id][node.id][1];
                ((((supply * demand) + ((supply - demand) * 5.0)) / 10.0)
                    * self.get_resource_demand_from_stockpiles(resource.clone()) as f32
                    * scalar)
                    + (supply
                        * self.get_resource_demand_from_processors(resource.clone()) as f32
                        * self.get_processordemandnavscalar())
            })
            .sum();

        let unitclass_salience = &root.globalsalience.unitclasssalience.read().unwrap();

        let unitclass_demand_value: f32 = ai
            .ship_cargo_attract
            .iter()
            .map(|(unitclassid, scalar)| {
                let attractive_unitclass = match unitclassid {
                    UnitClassID::ShipClass(shipclassid) => UnitClass::ShipClass(
                        root.shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                    UnitClassID::SquadronClass(squadronclassid) => UnitClass::SquadronClass(
                        root.squadronclasses
                            .iter()
                            .find(|squadronclass| squadronclass.id == squadronclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                };
                let demand = unitclass_salience[allegiance.id][unitclassid.get_index()][node.id][0];
                let supply = unitclass_salience[allegiance.id][unitclassid.get_index()][node.id][1];
                (demand - supply)
                    * self.get_unitclass_supply(attractive_unitclass.clone()) as f32
                    * scalar
            })
            .sum();
        //this checks how much value the node holds with regards to shipclasses the subject ship is seeking (to carry as cargo)
        let unitclass_supply_value: f32 = ai
            .ship_cargo_attract
            .iter()
            .map(|(unitclassid, scalar)| {
                //we index into the salience map by resource and then by node
                //to determine how much supply there is in this node for each resource the subject ship wants
                //NOTE: Previously, we got demand by indexing by nodeid, not location.
                //I believe using the ship's current position to calculate demand
                //will eliminate a pathology and produce more correct gradient-following behavior.
                let attractive_unitclass = match unitclassid {
                    UnitClassID::ShipClass(shipclassid) => UnitClass::ShipClass(
                        root.shipclasses
                            .iter()
                            .find(|shipclass| shipclass.id == shipclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                    UnitClassID::SquadronClass(squadronclassid) => UnitClass::SquadronClass(
                        root.squadronclasses
                            .iter()
                            .find(|squadronclass| squadronclass.id == squadronclassid.index)
                            .unwrap()
                            .clone(),
                    ),
                };
                let demand =
                    unitclass_salience[allegiance.id][unitclassid.get_index()][location.id][0];
                let supply = unitclass_salience[allegiance.id][unitclassid.get_index()][node.id][1];
                (((supply * demand) + ((supply - demand) * 5.0)) / 10.0)
                    * self.get_unitclass_demand(attractive_unitclass.clone()) as f32
                    * scalar
            })
            .sum();
        //this checks how much demand there is in the node for ships of the subject ship's class
        let ship_value_specific: f32 = unitclass_salience[allegiance.id]
            [self.get_unitclass().get_id()][node.id][0]
            * ai.ship_attract_specific;
        //oh, THIS is why we needed the placeholder ship class
        //this checks how much demand there is in the node for ships in general
        let ship_value_generic: f32 =
            unitclass_salience[allegiance.id][0][node.id][0] * ai.ship_attract_generic;

        let faction_salience = &root.globalsalience.factionsalience.read().unwrap();

        let faction_supply = transpose(&faction_salience[allegiance.id])[node.id]
            .iter()
            .map(|array| array[0])
            .sum::<f32>();
        let friendly_supply_value =
            faction_supply.clamp(0.0, f32::MAX) * ai.friendly_supply_attract;
        let hostile_supply_value = faction_supply.clamp(f32::MIN, 0.0) * ai.hostile_supply_attract;

        let allegiance_supply_unscaled = faction_salience[allegiance.id][allegiance.id][node.id][0];
        let allegiance_demand_value = (faction_salience[allegiance.id][allegiance.id][node.id][1]
            - allegiance_supply_unscaled)
            .clamp(0.0, f32::MAX)
            * ai.allegiance_demand_attract;

        let enemy_supply_unscaled = root
            .factions
            .iter()
            .filter(|faction| {
                root.wars.contains(&(
                    faction.clone().min(&allegiance.clone().clone()).clone(),
                    allegiance.clone().max(faction.clone().clone()).clone(),
                ))
            })
            .map(|faction| faction_salience[faction.id][faction.id][node.id][0])
            .sum::<f32>();
        let enemy_demand_unscaled = root
            .factions
            .iter()
            .filter(|faction| {
                root.wars.contains(&(
                    faction.clone().min(&allegiance.clone().clone()).clone(),
                    allegiance.clone().max(faction.clone().clone()).clone(),
                ))
            })
            .map(|faction| faction_salience[faction.id][faction.id][node.id][1])
            .sum::<f32>();
        let enemy_demand_value = (enemy_demand_unscaled - enemy_supply_unscaled)
            .clamp(0.0, f32::MAX)
            * ai.allegiance_demand_attract;

        NotNan::new(
            resource_demand_value
                + resource_supply_value
                + unitclass_demand_value
                + unitclass_supply_value
                + ship_value_specific
                + ship_value_generic
                + friendly_supply_value
                + hostile_supply_value
                + allegiance_demand_value
                + enemy_demand_value,
        )
        .unwrap()
    }
    fn navigate(
        //used for ships which are operating independently
        //this method determines which of the current node's neighbors is most desirable
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Arc<Node>> {
        let location = self.get_mother_node();
        //we iterate over the destinations to determine which neighbor is most desirable
        let neighbor_values: Vec<(Arc<Node>, NotNan<f32>)> = destinations
            .iter()
            //we go through all the different kinds of desirable salience values each node might have and add them up
            //then return the node with the highest value
            .map(|node| {
                //this checks how much value the node holds with regards to resources the subject ship is seeking
                (
                    node.clone(),
                    self.get_node_nav_attractiveness(root, node.clone()),
                )
            })
            .collect();
        let null_pair = (location.clone(), NotNan::new(0.0).unwrap());
        let best_neighbor = neighbor_values
            .iter()
            .max_by_key(|(_, val)| val)
            .unwrap_or(&null_pair);
        if best_neighbor.1
            > self.get_node_nav_attractiveness(root, location) * self.get_navthreshold()
        {
            Some(best_neighbor.0.clone())
        } else {
            None
        }
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>>;
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
                    if let Some(aggressor) =
                        root.engagement_check(destination.clone(), self.get_allegiance())
                    {
                        let engagement = root.internal_battle(EngagementPrep::engagement_prep(
                            root,
                            destination,
                            Some(aggressor),
                        ));
                        engagement.battle_cleanup(root);
                    }
                    self.deploy_daughters(root);
                }
                None => {}
            }
            destination_option
        } else {
            None
        }
    }
    fn deploy_daughters(&self, root: &Root) {
        if let Some(threshold) = self.get_deployment_threshold() {
            let daughters = self.get_daughters();
            let (active_daughters, passive_daughters): (Vec<&Unit>, Vec<&Unit>) =
                daughters.iter().partition(|daughter| {
                    daughter.get_deployment_status()
                        && (daughter.get_moves_left(root.turn.load(atomic::Ordering::Relaxed))
                            > threshold)
                });
            active_daughters.iter().for_each(|daughter| {
                daughter.transfer(UnitLocation::Node(self.get_mother_node()));
                let mut moving = true;
                while moving {
                    if (!daughter.is_in_node()) || daughter.maneuver(root).is_none() {
                        moving = false
                    }
                }
            });
            passive_daughters
                .iter()
                .for_each(|daughter| daughter.deploy_daughters(root));
        }
    }
    fn calculate_damage(
        &self,
        root: &Root,
        is_victor: bool,
        victor_strength: f32,
        victis_strength: f32,
        duration: u64,
        duration_damage_rand: f32,
        rng: &mut Hc128Rng,
    ) -> (u64, Vec<u64>);
    fn damage(&self, damage: u64, engine_damage: &Vec<u64>);
    fn repair(&self, per_engagement: bool);
    //Checks whether the unit will defect this turn; if it will, makes the unit defect and returns the node the ship is in afterward
    //so that that node can be checked for engagements
    fn try_defect(&self, root: &Root) -> Option<Vec<Arc<Node>>> {
        let location = self.get_mother_node();
        let allegiance = self.get_allegiance();
        let (defectchance, defectescapescalar) = self.get_defection_data();
        //NOTE: Test whether this gives us numbers that make sense.
        let local_threat_ratio: f32 = defectchance
            .iter()
            .map(|(faction, _)| {
                root.globalsalience.factionsalience.read().unwrap()[allegiance.id][faction.id]
                    [location.id][0]
            })
            .sum();
        //if defectchance only contains data for one faction
        //then either it's the faction it's currently part of, in which case we have no data on its chance of joining other factions
        //or it's not, in which case we have no data on its chance of defecting from its current faction
        //so either way we treat it as incapable of defecting
        let defect_probability = if defectchance.len() > 1 {
            ((local_threat_ratio * defectchance
                .get(&allegiance)
                .unwrap_or(&(0.0, 0.0))
                .0)
                //NOTE: I *think* that clamp will resolve things properly if we end up dividing by zero -- it'll set the probability to 1 -- but it's hard to be sure
                /self.get_morale_scalar())
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut rng = Hc128Rng::seed_from_u64(47);
        let defects = rng.gen_bool(defect_probability as f64);
        if defects {
            let interdiction: f32 = location
                .mutables
                .read()
                .unwrap()
                .units
                .iter()
                .filter(|unit| unit.get_allegiance() == allegiance)
                .filter(|unit| unit.get_id() != self.get_id())
                .map(|unit| unit.get_interdiction_scalar())
                .product();
            let new_faction_probabilities: Vec<(Arc<Faction>, f32)> = defectchance
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
            self.change_allegiance(new_faction.clone());
            //NOTE: This should take more things into account probably
            let escapes = rng.gen_bool((defectescapescalar / interdiction).clamp(0.0, 1.0) as f64);
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
                        Some(vec![location, destination])
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
    fn kill(&self);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipFlavor {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
}

impl PartialEq for ShipFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ShipFlavor {}

impl Ord for ShipFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for ShipFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for ShipFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ShipClassID {
    pub index: usize,
}

impl ShipClassID {
    pub fn new_from_index(index: usize) -> Self {
        ShipClassID { index: index }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShipClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub shipflavor: Arc<ShipFlavor>,
    pub basehull: u64,     //how many hull hitpoints this ship has by default
    pub basestrength: u64, //base strength score, used by AI to reason about ships' effectiveness; for an actual ship, this will be mutated based on current health and XP
    pub visibility: bool,
    pub propagates: bool,
    pub hangarvol: u64,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub defaultweapons: Option<HashMap<Arc<Resource>, u64>>, //a strikecraft's default weapons, which it always has with it
    pub hangars: Vec<Arc<HangarClass>>,
    pub engines: Vec<Arc<EngineClass>>,
    pub repairers: Vec<Arc<RepairerClass>>,
    pub internodeweapons: Vec<Arc<InterNodeWeaponClass>>,
    pub factoryclasslist: Vec<Arc<FactoryClass>>,
    pub shipyardclasslist: Vec<Arc<ShipyardClass>>,
    pub aiclass: Arc<ShipAI>,
    pub navthreshold: f32, //the value of an adjacent node must exceed (the value of the current node times navthreshold) in order for the ship to decide to move
    pub processordemandnavscalar: f32, //multiplier for demand generated by the ship's engines, repairers, factories, and shipyards, to modify it relative to that generated by stockpiles
    pub deploys_self: bool,            //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments; value is number of moves a daughter must be able to make to be deployed
    pub defectchance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub toughnessscalar: f32, //is used as a divisor for damage values taken by this ship in battle; a value of 2.0 will halve damage
    pub battleescapescalar: f32, //is added to toughnessscalar in battles where this ship is on the losing side, trying to escape
    pub defectescapescalar: f32, //influences how likely it is that a ship of this class will, if it defects, escape to an enemy-held node with no engagement taking place
    pub interdictionscalar: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this shipclass to be
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
    fn get_ideal_volume(&self) -> u64 {
        self.hangarvol
    }
    fn get_unitclass(class: Arc<Self>) -> UnitClass {
        UnitClass::ShipClass(class.clone())
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
                internodeweapons: class
                    .internodeweapons
                    .iter()
                    .map(|internodeweaponclass| {
                        InterNodeWeaponClass::instantiate(internodeweaponclass.clone())
                    })
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

#[derive(Debug, PartialEq, Clone)]
pub struct ShipInstanceMut {
    pub hull: u64, //how many hitpoints the ship has
    pub visibility: bool,
    pub stockpiles: Vec<PluripotentStockpile>,
    pub efficiency: f32,
    pub hangars: Vec<Arc<HangarInstance>>,
    pub engines: Vec<EngineInstance>,
    pub movement_left: u64, //starts at one trillion each turn, gets decremented as ship moves; represents time left to move during the turn
    pub repairers: Vec<RepairerInstance>,
    pub internodeweapons: Vec<Arc<InterNodeWeaponClass>>,
    pub factoryinstancelist: Vec<FactoryInstance>,
    pub shipyardinstancelist: Vec<ShipyardInstance>,
    pub location: UnitLocation, //where the ship is -- a node if it's unaffiliated, a squadron if it's in one
    pub allegiance: Arc<Faction>, //which faction this ship belongs to
    pub objectives: Vec<Objective>,
    pub aiclass: Arc<ShipAI>,
}

#[derive(Debug)]
pub struct ShipInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: Arc<ShipClass>, //which class of ship this is
    pub mutables: RwLock<ShipInstanceMut>,
}

impl ShipInstance {
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

impl PartialEq for ShipInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.visiblename == other.visiblename
            && self.class == other.class
            && self.mutables.read().unwrap().clone() == other.mutables.read().unwrap().clone()
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
    fn get_unit(&self) -> Unit {
        Unit::Ship(self.clone())
    }
    fn get_unitclass(&self) -> UnitClass {
        UnitClass::ShipClass(self.class.clone())
    }
    fn get_ship(&self) -> Option<Arc<ShipInstance>> {
        Some(self.clone())
    }
    fn get_squadron(&self) -> Option<Arc<SquadronInstance>> {
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
            UnitLocation::Squadron(squadron) => squadron
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
    fn get_hull(&self) -> u64 {
        self.mutables.read().unwrap().hull
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
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
    fn get_daughters_recursive(&self) -> Vec<Unit> {
        iter::once(self.clone().get_unit())
            .chain(
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
                            .map(|unit| unit.get_daughters_recursive())
                            .collect::<Vec<Vec<Unit>>>()
                    })
                    .flatten()
                    .flatten(),
            )
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
    fn get_interdiction_scalar(&self) -> f32 {
        self.class.interdictionscalar
            * self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_interdiction_scalar())
                .product::<f32>()
    }
    fn get_processordemandnavscalar(&self) -> f32 {
        self.class.processordemandnavscalar
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
    //this one is used for checking which faction has the most strength left after a battle
    //we know how much damage the ship will take, but it hasn't actually been applied yet
    //also, we don't worry about daughters because we're evaluating all units separately
    fn get_strength_post_engagement(&self, damage: u64) -> u64 {
        let mutables = self.mutables.read().unwrap();
        let objective_strength: f32 = mutables
            .objectives
            .iter()
            .map(|of| of.get_scalars().strengthscalar)
            .product();
        (self.class.basestrength as f32
            * ((mutables.hull - damage) as f32 / self.class.basehull as f32)
            * self.get_character_strength_scalar()
            * objective_strength) as u64
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
            friendly_supply_attract: ai.friendly_supply_attract.clone(),
            hostile_supply_attract: ai.hostile_supply_attract.clone(),
            allegiance_demand_attract: ai.allegiance_demand_attract.clone(),
            enemy_demand_attract: ai.enemy_demand_attract.clone(),
        }
    }
    fn get_navthreshold(&self) -> f32 {
        self.class.navthreshold.clone()
    }
    fn get_objectives(&self) -> Vec<Objective> {
        self.mutables
            .read()
            .unwrap()
            .objectives
            .iter()
            .cloned()
            .collect()
    }
    fn get_deployment_threshold(&self) -> Option<u64> {
        self.class.deploys_daughters
    }
    fn get_deployment_status(&self) -> bool {
        self.class.deploys_self
    }
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32) {
        (
            self.class.defectchance.clone(),
            self.class.defectescapescalar,
        )
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>()
            + mutables
                .factoryinstancelist
                .iter()
                .map(|f| f.get_resource_supply_total(resource.clone()))
                .sum::<u64>()
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_supply(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagates)
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
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_demand(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
            .stockpiles
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_demand(resource.clone()))
            .sum::<u64>()
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_demand_from_stockpiles(resource.clone()))
                .sum::<u64>()
    }
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64 {
        let mutables = self.mutables.read().unwrap();
        mutables
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
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_resource_demand_from_processors(resource.clone()))
                .sum::<u64>()
    }
    //NOTE: Here in get_unitclass_num/unitclass_supply, we use an approach where we count the ship itself if it matches the subject unitclass.
    //Therefore, we don't need to check how many of an entity's direct daughters match the class alongside feeding the method down to them;
    //we can just give the daughters (or the hangars) the method.
    //This allows for more elegant recursion, but may have disadvantages.
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        (&self.get_unitclass() == &unitclass) as u64
            + self
                .mutables
                .read()
                .unwrap()
                .hangars
                .iter()
                .filter(|hangar| hangar.class.propagates)
                .map(|hangar| hangar.get_unitclass_num(unitclass.clone()))
                .sum::<u64>()
    }
    //NOTE: It's possible that this approach of gathering the volume data one ship at a time like this is less performant than filtering down the collection of units,
    //getting length, then multiplying by volume
    fn get_unitclass_supply(&self, unitclass: UnitClass) -> u64 {
        ((&self.get_unitclass() == &unitclass) as u64 * self.get_volume())
            + self
                .mutables
                .read()
                .unwrap()
                .hangars
                .iter()
                .filter(|hangar| hangar.class.propagates)
                .map(|hangar| hangar.get_unitclass_supply(unitclass.clone()))
                .sum::<u64>()
    }
    fn get_unitclass_demand(&self, unitclass: UnitClass) -> u64 {
        self.mutables
            .read()
            .unwrap()
            .hangars
            .iter()
            .filter(|hangar| hangar.class.propagates)
            .map(|hangar| hangar.get_unitclass_demand(unitclass.clone()))
            .sum::<u64>()
    }
    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        self.mutables.write().unwrap().allegiance = new_faction.clone();
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.change_allegiance(new_faction.clone()));
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        let source = self.get_location();
        if source.check_remove(self.get_unit())
            && destination.check_insert(self.get_unit())
            && self.acyclicity_check(destination.clone())
        {
            source.remove_unit(self.get_unit());
            //NOTE: Make sure cloning destination here clones the arc rather than cloning the thing inside the arc
            self.mutables.write().unwrap().location = destination.clone();
            destination.insert_unit(self.get_unit());
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
    fn set_movement_recursive(&self, value: u64) {
        let mut mutables = self.mutables.write().unwrap();
        let new_value = mutables.movement_left.saturating_sub(value);
        mutables.movement_left = new_value;
        drop(mutables);
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.set_movement_recursive(value));
    }
    fn get_moves_left(&self, turn: u64) -> u64 {
        if let Some(engine) = self
            .mutables
            .read()
            .unwrap()
            .engines
            .iter()
            .find(|engine| engine.check_engine_movement_only(turn))
        {
            engine.get_moves_left(self.mutables.read().unwrap().movement_left)
        } else {
            0
        }
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        let location = self.get_mother_node();
        let mut mutables = self.mutables.write().unwrap();
        let movement_left_old = mutables.movement_left;
        match movement_left_old > 0 {
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
        let movement_left = mutables.movement_left;
        drop(mutables);
        self.set_movement_recursive(movement_left.saturating_sub(movement_left_old));
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        self.process_engines(root, destination.clone());
        self.transfer(UnitLocation::Node(destination.clone()));
        Some(destination)
    }
    fn calculate_damage(
        &self,
        root: &Root,
        is_victor: bool,
        allied_strength: f32,
        enemy_strength: f32,
        duration: u64,
        duration_damage_rand: f32,
        rng: &mut Hc128Rng,
    ) -> (u64, Vec<u64>) {
        //to calculate how much damage the ship takes
        //we first have a multiplicative damage value, which is:
        //the ship's maximum health
        //times the ratio between the enemy's strength and the ship's coalition's strength
        //times the battle's duration as a fraction of the 'typical' battle duration, modified by a modder-specified scalar and a battle-wide random value
        //times a ship-specific random factor
        //times the modder-specified multiplier for damage taken by winning or losing ships
        //
        //then we add to that an additive damage value, which is:
        //the modder-defined base damage value
        //times the strength ratio
        //times the duration modifier
        //times the random factor
        //times the losing-ship multiplier
        //
        //then we divide all that by the sum of the ship's toughness and escape scalars
        let rand_factor = Normal::<f32>::new(0.25, root.config.battlescalars.damage_dev)
            .unwrap()
            .sample(rng)
            .clamp(0.0, 10.0);
        let vae = match is_victor {
            true => root.config.battlescalars.vae_victor,
            false => root.config.battlescalars.vae_victis,
        };
        //we do basically the same thing for winning ships and losing ships
        //except that the strength ratio is reversed
        //we use the damage multiplier for winners or losers
        //and we don't take battleescapescalar into account for winners
        let damage = (((self.class.basehull as f32
            * (enemy_strength / allied_strength)
            * ((duration as f32 / root.config.battlescalars.avg_duration as f32)
                * root.config.battlescalars.duration_damage_scalar
                * duration_damage_rand)
            * rand_factor
            * vae)
            + (root.config.battlescalars.base_damage
                * (enemy_strength / allied_strength)
                * ((duration as f32 / root.config.battlescalars.avg_duration as f32)
                    * root.config.battlescalars.duration_damage_scalar
                    * duration_damage_rand)
                * rand_factor
                * vae))
            / (self.class.toughnessscalar
                + (self.class.battleescapescalar * !is_victor as i8 as f32)))
            as u64;
        let engine_damage: Vec<u64> = self
            .mutables
            .read()
            .unwrap()
            .engines
            .iter()
            .filter(|e| e.health.is_some())
            .map(|e| {
                ((damage as f32
                    * Normal::<f32>::new(1.0, root.config.battlescalars.damage_dev)
                        .unwrap()
                        .sample(rng)
                        .clamp(0.0, 2.0)
                    * root.config.battlescalars.engine_damage_scalar)
                    / e.toughnessscalar) as u64
            })
            .collect();
        (damage, engine_damage)
    }
    fn damage(&self, damage: u64, engine_damage: &Vec<u64>) {
        let mut mutables = self.mutables.write().unwrap();
        mutables.hull = mutables.hull.saturating_sub(damage);
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
    fn repair(&self, per_engagement: bool) {
        let mut mutables: RwLockWriteGuard<ShipInstanceMut> = self.mutables.write().unwrap();
        let current_hull = mutables.hull;
        if current_hull < self.class.basehull && current_hull > 0
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
    fn kill(&self) {
        self.mutables.write().unwrap().hull = 0;
        self.get_daughters().iter().for_each(|ship| ship.kill());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquadronFlavor {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
}

impl PartialEq for SquadronFlavor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SquadronFlavor {}

impl Ord for SquadronFlavor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for SquadronFlavor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for SquadronFlavor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SquadronClassID {
    pub index: usize,
}

impl SquadronClassID {
    pub fn new_from_index(index: usize) -> Self {
        SquadronClassID { index: index }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SquadronClass {
    pub id: usize,
    pub visiblename: String,
    pub description: String,
    pub squadronflavor: Arc<SquadronFlavor>,
    pub visibility: bool,
    pub propagates: bool,
    pub strengthmod: (f32, u64),
    pub squadronconfig: HashMap<UnitClassID, u64>,
    pub non_ideal_supply_scalar: f32, //multiplier used for demand generated by non-ideal ships under the target limit; should be below one
    pub target: u64,
    pub navthreshold: f32, //the value of an adjacent node must exceed (the value of the current node times navthreshold) in order for the ship to decide to move
    pub navquorum: f32,
    pub disbandthreshold: f32,
    pub deploys_self: bool, //if false, ship will not go on deployments
    pub deploys_daughters: Option<u64>, // if None, ship will not send its daughters on deployments
    pub defectchance: HashMap<Arc<Faction>, (f32, f32)>, //first number is probability scalar for defection *from* the associated faction; second is scalar for defection *to* it
    pub defectescapescalar: f32,
    pub value_mult: f32, //how valuable the AI considers one volume point of this squadronclass to be
}

impl SquadronClass {
    pub fn get_ideal_strength(&self, root: &Root) -> u64 {
        self.squadronconfig
            .iter()
            .map(|(unitclassid, v)| unitclassid.get_unitclass(root).get_ideal_strength(root) * v)
            .sum()
    }
    pub fn get_ideal_volume(&self) -> u64 {
        self.target
    }
    fn get_unitclass(class: Arc<Self>) -> UnitClass {
        UnitClass::SquadronClass(class.clone())
    }
    pub fn instantiate(
        class: Arc<Self>,
        location: Arc<Node>,
        faction: Arc<Faction>,
        root: &Root,
    ) -> SquadronInstance {
        let index = root.unitcounter.fetch_add(1, atomic::Ordering::Relaxed);
        SquadronInstance {
            id: index,
            visiblename: uuid::Uuid::new_v4().to_string(),
            class: class.clone(),
            idealstrength: class.get_ideal_strength(root),
            mutables: RwLock::new(SquadronInstanceMut {
                visibility: class.visibility,
                location: UnitLocation::Node(location),
                daughters: Vec::new(),
                allegiance: faction,
                objectives: Vec::new(),
                ghost: true,
            }),
        }
    }
}

impl Eq for SquadronClass {}

impl Ord for SquadronClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for SquadronClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for SquadronClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

//NOTE: At present, this is an exact copy of ShipAI except that it doesn't have an ID, because it's always calculated on the fly.
#[derive(Debug, Clone)]
pub struct NavAI {
    pub ship_attract_specific: f32,
    pub ship_attract_generic: f32,
    pub ship_cargo_attract: HashMap<UnitClassID, f32>,
    pub resource_attract: HashMap<Arc<Resource>, f32>,
    pub friendly_supply_attract: f32,
    pub hostile_supply_attract: f32,
    pub allegiance_demand_attract: f32,
    pub enemy_demand_attract: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SquadronInstanceMut {
    pub visibility: bool,
    pub location: UnitLocation,
    pub daughters: Vec<Unit>,
    pub allegiance: Arc<Faction>,
    pub objectives: Vec<Objective>,
    pub ghost: bool,
}

#[derive(Debug)]
pub struct SquadronInstance {
    pub id: u64,
    pub visiblename: String,
    pub class: Arc<SquadronClass>,
    pub idealstrength: u64,
    pub mutables: RwLock<SquadronInstanceMut>,
}

impl PartialEq for SquadronInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.visiblename == other.visiblename
            && self.class == other.class
            && self.idealstrength == other.idealstrength
            && self.mutables.read().unwrap().clone() == other.mutables.read().unwrap().clone()
    }
}

impl Eq for SquadronInstance {}

impl Ord for SquadronInstance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for SquadronInstance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for SquadronInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Mobility for Arc<SquadronInstance> {
    fn get_unit(&self) -> Unit {
        Unit::Squadron(self.clone())
    }
    fn get_unitclass(&self) -> UnitClass {
        UnitClass::SquadronClass(self.class.clone())
    }
    fn get_ship(&self) -> Option<Arc<ShipInstance>> {
        None
    }
    fn get_squadron(&self) -> Option<Arc<SquadronInstance>> {
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
            UnitLocation::Squadron(squadron) => squadron
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
    fn get_hull(&self) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_hull())
            .sum()
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
    fn get_daughters_recursive(&self) -> Vec<Unit> {
        iter::once(self.clone().get_unit())
            .chain(
                self.mutables
                    .read()
                    .unwrap()
                    .daughters
                    .iter()
                    .map(|daughter| daughter.get_daughters_recursive())
                    .flatten(),
            )
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
    fn get_interdiction_scalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_interdiction_scalar())
            .product()
    }
    fn get_processordemandnavscalar(&self) -> f32 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_processordemandnavscalar())
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
    fn get_strength_post_engagement(&self, _damage: u64) -> u64 {
        0
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
                friendly_supply_attract: 1.0,
                hostile_supply_attract: 1.0,
                allegiance_demand_attract: 1.0,
                enemy_demand_attract: 1.0,
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
    fn get_navthreshold(&self) -> f32 {
        self.class.navthreshold.clone()
    }
    fn get_objectives(&self) -> Vec<Objective> {
        self.mutables
            .read()
            .unwrap()
            .objectives
            .iter()
            .cloned()
            .collect()
    }
    fn get_deployment_threshold(&self) -> Option<u64> {
        self.class.deploys_daughters
    }
    fn get_deployment_status(&self) -> bool {
        self.class.deploys_self
    }
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32) {
        (
            self.class.defectchance.clone(),
            self.class.defectescapescalar,
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
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand_from_stockpiles(resource.clone()))
            .sum()
    }
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_resource_demand_from_processors(resource.clone()))
            .sum()
    }
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        (&self.get_unitclass() == &unitclass) as u64
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_unitclass_num(unitclass.clone()))
                .sum::<u64>()
    }
    fn get_unitclass_supply(&self, unitclass: UnitClass) -> u64 {
        let daughter_supply = self
            .get_daughters()
            .iter()
            .map(|daughter| daughter.get_unitclass_supply(unitclass.clone()))
            .sum::<u64>();
        let ideal_volume = self
            .class
            .squadronconfig
            .get(&UnitClassID::new_from_unitclass(&unitclass))
            .unwrap_or(&0)
            * unitclass.get_ideal_volume();
        let non_ideal_volume = daughter_supply.saturating_sub(ideal_volume);
        let excess_volume = self.get_volume().saturating_sub(self.class.target);
        let over_target_supply = (excess_volume).min(non_ideal_volume);
        let under_target_supply = ((non_ideal_volume.saturating_sub(over_target_supply)) as f32
            * self.class.non_ideal_supply_scalar) as u64;
        ((&self.get_unitclass() == &unitclass) as u64 * self.get_volume())
            + over_target_supply
            + under_target_supply
    }
    fn get_unitclass_demand(&self, unitclass: UnitClass) -> u64 {
        let daughter_volume = self
            .get_daughters()
            .iter()
            .filter(|unit| &unit.get_unitclass() == &unitclass)
            .map(|unit| unit.get_volume())
            .sum::<u64>();
        let ideal_volume = self
            .class
            .squadronconfig
            .get(&UnitClassID::new_from_unitclass(&unitclass))
            .unwrap_or(&0)
            * unitclass.get_ideal_volume();
        ideal_volume.saturating_sub(daughter_volume)
            + self
                .get_daughters()
                .iter()
                .map(|daughter| daughter.get_unitclass_demand(unitclass.clone()))
                .sum::<u64>()
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
                            .push(self.get_unit());
                        true
                    }
                    UnitLocation::Squadron(_) => false,
                    UnitLocation::Hangar(_) => false,
                }
            }
            UnitLocation::Squadron(_) => false,
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
        //we see what fraction of the squadron's strength is unable to make the jump
        //by checking strength of failed ships, and then all daughters
        //we don't just call get_strength on the squadron itself
        //if we did, the squadron's strength modifiers would be counted only toward its total
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
    fn set_movement_recursive(&self, value: u64) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.set_movement_recursive(value));
    }
    fn get_moves_left(&self, turn: u64) -> u64 {
        self.get_daughters()
            .iter()
            .map(|daughter| daughter.get_moves_left(turn))
            .min()
            .unwrap_or(0)
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.process_engines(root, destination.clone()));
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
    fn calculate_damage(
        &self,
        _root: &Root,
        _is_victor: bool,
        _victor_strength: f32,
        _victis_strength: f32,
        _duration: u64,
        _duration_damage_rand: f32,
        _rng: &mut Hc128Rng,
    ) -> (u64, Vec<u64>) {
        (0, Vec::new())
    }
    fn damage(&self, _damage: u64, _engine_damage: &Vec<u64>) {}
    fn repair(&self, _per_engagement: bool) {}
    fn kill(&self) {
        self.get_daughters()
            .iter()
            .for_each(|daughter| daughter.kill());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum UnitClassID {
    ShipClass(ShipClassID),
    SquadronClass(SquadronClassID),
}

impl UnitClassID {
    fn new_from_unitclass(unitclass: &UnitClass) -> Self {
        match unitclass {
            UnitClass::ShipClass(sc) => UnitClassID::ShipClass(ShipClassID::new_from_index(sc.id)),
            UnitClass::SquadronClass(fc) => {
                UnitClassID::SquadronClass(SquadronClassID::new_from_index(fc.id))
            }
        }
    }
    fn get_index(&self) -> usize {
        match self {
            UnitClassID::ShipClass(sc) => sc.index,
            UnitClassID::SquadronClass(fc) => fc.index,
        }
    }
    fn get_unitclass(&self, root: &Root) -> UnitClass {
        match self {
            UnitClassID::ShipClass(sc) => UnitClass::ShipClass(
                root.shipclasses
                    .iter()
                    .find(|shipclass| shipclass.id == sc.index)
                    .unwrap()
                    .clone(),
            ),
            UnitClassID::SquadronClass(fc) => UnitClass::SquadronClass(
                root.squadronclasses
                    .iter()
                    .find(|squadronclass| squadronclass.id == fc.index)
                    .unwrap()
                    .clone(),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum UnitClass {
    ShipClass(Arc<ShipClass>),
    SquadronClass(Arc<SquadronClass>),
}

impl UnitClass {
    fn get_id(&self) -> usize {
        match self {
            UnitClass::ShipClass(sc) => sc.id,
            UnitClass::SquadronClass(fc) => fc.id,
        }
    }
    fn get_ideal_strength(&self, root: &Root) -> u64 {
        match self {
            UnitClass::ShipClass(sc) => sc.get_ideal_strength(root),
            UnitClass::SquadronClass(fc) => fc.get_ideal_strength(root),
        }
    }
    fn get_ideal_volume(&self) -> u64 {
        match self {
            UnitClass::ShipClass(sc) => sc.get_ideal_volume(),
            UnitClass::SquadronClass(fc) => fc.get_ideal_volume(),
        }
    }
    fn get_value_mult(&self) -> f32 {
        match self {
            UnitClass::ShipClass(sc) => sc.value_mult,
            UnitClass::SquadronClass(fc) => fc.value_mult,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Unit {
    Ship(Arc<ShipInstance>),
    Squadron(Arc<SquadronInstance>),
}

impl Mobility for Unit {
    fn get_unit(&self) -> Unit {
        self.clone()
    }
    fn get_unitclass(&self) -> UnitClass {
        match self {
            Unit::Ship(ship) => ship.get_unitclass(),
            Unit::Squadron(squadron) => squadron.get_unitclass(),
        }
    }
    fn get_ship(&self) -> Option<Arc<ShipInstance>> {
        match self {
            Unit::Ship(ship) => Some(ship.clone()),
            _ => None,
        }
    }
    fn get_squadron(&self) -> Option<Arc<SquadronInstance>> {
        match self {
            Unit::Squadron(squadron) => Some(squadron.clone()),
            _ => None,
        }
    }
    fn get_id(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_id(),
            Unit::Squadron(squadron) => squadron.get_id(),
        }
    }
    fn is_ship(&self) -> bool {
        match self {
            Unit::Ship(ship) => ship.is_ship(),
            Unit::Squadron(squadron) => squadron.is_ship(),
        }
    }
    fn get_location(&self) -> UnitLocation {
        match self {
            Unit::Ship(ship) => ship.get_location(),
            Unit::Squadron(squadron) => squadron.get_location(),
        }
    }
    fn check_location_coherency(&self) {
        match self {
            Unit::Ship(ship) => ship.check_location_coherency(),
            Unit::Squadron(squadron) => squadron.check_location_coherency(),
        }
    }
    fn get_hull(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_hull(),
            Unit::Squadron(squadron) => squadron.get_hull(),
        }
    }
    fn get_allegiance(&self) -> Arc<Faction> {
        match self {
            Unit::Ship(ship) => ship.get_allegiance(),
            Unit::Squadron(squadron) => squadron.get_allegiance(),
        }
    }
    fn get_daughters(&self) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_daughters(),
            Unit::Squadron(squadron) => squadron.get_daughters(),
        }
    }
    fn get_daughters_recursive(&self) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_daughters_recursive(),
            Unit::Squadron(squadron) => squadron.get_daughters_recursive(),
        }
    }
    fn get_morale_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_morale_scalar(),
            Unit::Squadron(squadron) => squadron.get_morale_scalar(),
        }
    }
    fn get_character_strength_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_character_strength_scalar(),
            Unit::Squadron(squadron) => squadron.get_character_strength_scalar(),
        }
    }
    fn get_interdiction_scalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_interdiction_scalar(),
            Unit::Squadron(squadron) => squadron.get_interdiction_scalar(),
        }
    }
    fn get_processordemandnavscalar(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_processordemandnavscalar(),
            Unit::Squadron(squadron) => squadron.get_processordemandnavscalar(),
        }
    }
    fn get_strength(&self, time: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength(time),
            Unit::Squadron(squadron) => squadron.get_strength(time),
        }
    }
    fn get_strength_post_engagement(&self, damage: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_strength_post_engagement(damage),
            Unit::Squadron(squadron) => squadron.get_strength_post_engagement(damage),
        }
    }
    fn get_volume(&self) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_volume(),
            Unit::Squadron(squadron) => squadron.get_volume(),
        }
    }
    fn get_ai(&self) -> NavAI {
        match self {
            Unit::Ship(ship) => ship.get_ai(),
            Unit::Squadron(squadron) => squadron.get_ai(),
        }
    }
    fn get_navthreshold(&self) -> f32 {
        match self {
            Unit::Ship(ship) => ship.get_navthreshold(),
            Unit::Squadron(squadron) => squadron.get_navthreshold(),
        }
    }
    fn get_objectives(&self) -> Vec<Objective> {
        match self {
            Unit::Ship(ship) => ship.get_objectives(),
            Unit::Squadron(squadron) => squadron.get_objectives(),
        }
    }
    fn get_deployment_threshold(&self) -> Option<u64> {
        match self {
            Unit::Ship(ship) => ship.get_deployment_threshold(),
            Unit::Squadron(squadron) => squadron.get_deployment_threshold(),
        }
    }
    fn get_deployment_status(&self) -> bool {
        match self {
            Unit::Ship(ship) => ship.get_deployment_status(),
            Unit::Squadron(squadron) => squadron.get_deployment_status(),
        }
    }
    fn get_defection_data(&self) -> (HashMap<Arc<Faction>, (f32, f32)>, f32) {
        match self {
            Unit::Ship(ship) => ship.get_defection_data(),
            Unit::Squadron(squadron) => squadron.get_defection_data(),
        }
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_supply(resource),
            Unit::Squadron(squadron) => squadron.get_resource_supply(resource),
        }
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand(resource),
            Unit::Squadron(squadron) => squadron.get_resource_demand(resource),
        }
    }
    fn get_resource_demand_from_stockpiles(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand_from_stockpiles(resource),
            Unit::Squadron(squadron) => squadron.get_resource_demand_from_stockpiles(resource),
        }
    }
    fn get_resource_demand_from_processors(&self, resource: Arc<Resource>) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_resource_demand_from_processors(resource),
            Unit::Squadron(squadron) => squadron.get_resource_demand_from_processors(resource),
        }
    }
    fn get_unitclass_num(&self, unitclass: UnitClass) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_unitclass_num(unitclass),
            Unit::Squadron(squadron) => squadron.get_unitclass_num(unitclass),
        }
    }
    fn get_unitclass_supply(&self, unitclass: UnitClass) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_unitclass_supply(unitclass),
            Unit::Squadron(squadron) => squadron.get_unitclass_supply(unitclass),
        }
    }
    fn get_unitclass_demand(&self, unitclass: UnitClass) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_unitclass_demand(unitclass),
            Unit::Squadron(squadron) => squadron.get_unitclass_demand(unitclass),
        }
    }

    fn change_allegiance(&self, new_faction: Arc<Faction>) {
        match self {
            Unit::Ship(ship) => ship.change_allegiance(new_faction),
            Unit::Squadron(squadron) => squadron.change_allegiance(new_faction),
        }
    }
    fn transfer(&self, destination: UnitLocation) -> bool {
        match self {
            Unit::Ship(ship) => ship.transfer(destination),
            Unit::Squadron(squadron) => squadron.transfer(destination),
        }
    }
    fn destinations_check(
        &self,
        root: &Root,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<Vec<Arc<Node>>> {
        match self {
            Unit::Ship(ship) => ship.destinations_check(root, destinations),
            Unit::Squadron(squadron) => squadron.destinations_check(root, destinations),
        }
    }
    fn get_traversal_checked_daughters(&self, root: &Root, destination: Arc<Node>) -> Vec<Unit> {
        match self {
            Unit::Ship(ship) => ship.get_traversal_checked_daughters(root, destination),
            Unit::Squadron(squadron) => squadron.get_traversal_checked_daughters(root, destination),
        }
    }
    fn set_movement_recursive(&self, value: u64) {
        match self {
            Unit::Ship(ship) => ship.set_movement_recursive(value),
            Unit::Squadron(squadron) => squadron.set_movement_recursive(value),
        }
    }
    fn get_moves_left(&self, turn: u64) -> u64 {
        match self {
            Unit::Ship(ship) => ship.get_moves_left(turn),
            Unit::Squadron(squadron) => squadron.get_moves_left(turn),
        }
    }
    fn process_engines(&self, root: &Root, destination: Arc<Node>) {
        match self {
            Unit::Ship(ship) => ship.process_engines(root, destination),
            Unit::Squadron(squadron) => squadron.process_engines(root, destination),
        }
    }
    fn traverse(&self, root: &Root, destination: Arc<Node>) -> Option<Arc<Node>> {
        match self {
            Unit::Ship(ship) => ship.traverse(root, destination),
            Unit::Squadron(squadron) => squadron.traverse(root, destination),
        }
    }
    fn calculate_damage(
        &self,
        root: &Root,
        is_victor: bool,
        victor_strength: f32,
        victis_strength: f32,
        duration: u64,
        duration_damage_rand: f32,
        rng: &mut Hc128Rng,
    ) -> (u64, Vec<u64>) {
        match self {
            Unit::Ship(ship) => ship.calculate_damage(
                root,
                is_victor,
                victor_strength,
                victis_strength,
                duration,
                duration_damage_rand,
                rng,
            ),
            Unit::Squadron(squadron) => squadron.calculate_damage(
                root,
                is_victor,
                victor_strength,
                victis_strength,
                duration,
                duration_damage_rand,
                rng,
            ),
        }
    }
    fn damage(&self, damage: u64, engine_damage: &Vec<u64>) {
        match self {
            Unit::Ship(ship) => ship.damage(damage, engine_damage),
            Unit::Squadron(squadron) => squadron.damage(damage, engine_damage),
        }
    }
    fn repair(&self, per_engagement: bool) {
        match self {
            Unit::Ship(ship) => ship.repair(per_engagement),
            Unit::Squadron(squadron) => squadron.repair(per_engagement),
        }
    }
    fn kill(&self) {
        match self {
            Unit::Ship(ship) => ship.kill(),
            Unit::Squadron(squadron) => squadron.kill(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveScalars {
    pub difficulty: f32,
    pub cost: u64,
    pub durationscalar: f32,
    pub strengthscalar: f32,
    pub toughnessscalar: f32,
    pub battleescapescalar: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Objective {
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
    SquadronDeath {
        scalars: ObjectiveScalars,
        squadron: Arc<SquadronInstance>,
    },
    SquadronSafe {
        scalars: ObjectiveScalars,
        squadron: Arc<SquadronInstance>,
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

impl Objective {
    pub fn get_scalars(&self) -> ObjectiveScalars {
        match self {
            Objective::ReachNode { scalars, .. } => *scalars,
            Objective::ShipDeath { scalars, .. } => *scalars,
            Objective::ShipSafe { scalars, .. } => *scalars,
            Objective::SquadronDeath { scalars, .. } => *scalars,
            Objective::SquadronSafe { scalars, .. } => *scalars,
            Objective::NodeCapture { scalars, .. } => *scalars,
            Objective::NodeSafe { scalars, .. } => *scalars,
            Objective::SystemCapture { scalars, .. } => *scalars,
            Objective::SystemSafe { scalars, .. } => *scalars,
        }
    }
}

#[derive(Debug)]
pub struct Operation {
    pub visiblename: String,
    pub objectives: Vec<Objective>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FactionForces {
    pub local_forces: Vec<Unit>,
    pub reinforcements: Vec<(u64, Vec<Unit>)>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct UnitStatus {
    pub location: Option<UnitLocation>,
    pub damage: u64,
    pub engine_damage: Vec<u64>,
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
    fn calculate_engagement_duration(&self, root: &Root, rng: &mut Hc128Rng) -> u64 {
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
                                    unit.get_strength(root.config.battlescalars.avg_duration)
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
                                    .map(|objective| objective.get_scalars().durationscalar)
                                    .product::<f32>() //we multiply the individual objectives' scalars
                            })
                            .product::<f32>() //then the units'
                    })
                    .product::<f32>() //then the factions'
            })
            .product::<f32>(); //and finally the coalitions'

        let duration: u64 = (((battle_size as f32).log(root.config.battlescalars.duration_log_exp)
            + 300.0)
            * objective_duration_scalar
            * Normal::new(1.0, root.config.battlescalars.duration_dev)
                .unwrap()
                .sample(rng))
        .clamp(0.0, 2.0) as u64;
        duration
    }
    fn get_coalition_strengths(&self, duration: u64) -> HashMap<u64, u64> {
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
                                .map(|(lag, units)| {
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
    fn get_coalition_objective_difficulties(&self) -> HashMap<u64, f32> {
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
                                        .map(|objective| objective.get_scalars().difficulty)
                                        .product::<f32>()
                                })
                                .product::<f32>()
                        })
                        .product(),
                )
            })
            .collect()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Engagement {
    pub visiblename: String,
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
        println!("{}", self.visiblename);
        self.location.mutables.write().unwrap().allegiance = self.victors.0.clone();
        self.unit_status.iter().for_each(|(_, faction_map)| {
            faction_map.iter().for_each(|(_, unit_map)| {
                unit_map.iter().for_each(|(unit, status)| {
                    if let Some(place) = &status.location {
                        unit.damage(status.damage, &status.engine_damage);
                        unit.transfer(place.clone());
                        unit.repair(true);
                    } else {
                        unit.kill();
                    }
                })
            })
        });
        root.remove_dead();
        root.disband_squadrons();
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
    //this retrieves the value of a specific salience in a specific node
    fn get_value(
        self,
        root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        battle_duration: u64,
    ) -> Option<f32>;
}

//this method retrieves threat value generated by a given faction
impl Salience<polarity::Supply> for Arc<Faction> {
    fn get_value(
        self,
        _root: &Root,
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
            .map(|strength| strength as f32 * relation * self.value_mult)
    }
}

impl Salience<polarity::Demand> for Arc<Faction> {
    fn get_value(
        self,
        root: &Root,
        node: Arc<Node>,
        faction: Arc<Faction>,
        _battle_duration: u64,
    ) -> Option<f32> {
        //Here, we don't take the object faction's relations with the subject faction into account at all,
        //because subjectivity isn't really relevant for threat demand.
        //NOTE: Doing this by retrieving the data from salience state (rather than by looking at assets in the node directly) means that a node will want friendly forces corresponding to assets in other nodes
        //which are propagating salience into it. Not sure if this is desirable behavior.
        if faction == self {
            let resource_salience_by_node: Vec<Vec<[f32; 2]>> =
                transpose(&root.globalsalience.resourcesalience.read().unwrap()[self.id]);
            let resource_supply: f32 = resource_salience_by_node[node.id]
                .iter()
                .map(|array| array[0])
                .sum();
            let unitclass_salience_by_node: Vec<Vec<[f32; 2]>> =
                transpose(&root.globalsalience.unitclasssalience.read().unwrap()[self.id]);
            let unitclass_supply: f32 = unitclass_salience_by_node[node.id]
                .iter()
                .map(|array| array[0])
                .sum();
            let node_value = resource_supply + unitclass_supply;
            let node_object_faction_supply =
                root.globalsalience.factionsalience.read().unwrap()[self.id][self.id][node.id][0];
            Some(
                ((node_value
                    * self.volume_strength_ratio
                    * root.config.saliencescalars.volume_strength_ratio)
                    - node_object_faction_supply.clamp(0.0, f32::MAX))
                    * self.value_mult,
            )
        } else {
            None
        }
    }
}

//this method tells us how much supply there is of a given resource in a given node
impl Salience<polarity::Supply> for Arc<Resource> {
    fn get_value(
        self,
        _root: &Root,
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
        //then all the valid resource quantity in units
        let shipsupply: u64 = node
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_resource_supply(self.clone()))
            .sum::<u64>();
        //then sum them together
        let sum = (factorysupply + shipsupply) as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.value_mult)
        }
    }
}

//this method tells us how much demand there is for a given resource in a given node
impl Salience<polarity::Demand> for Arc<Resource> {
    fn get_value(
        self,
        _root: &Root,
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
        //now we have to look at units in the node, since they might have stockpiles of their own
        let shipdemand: u64 = node
            .mutables
            .read()
            .unwrap()
            .units
            .iter()
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_resource_demand(self.clone()))
            .sum::<u64>();
        //and sum everything together
        let sum = (factorydemand + shipyarddemand + shipdemand) as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.value_mult)
        }
    }
}

//this method tells us how much supply there is of a given shipclass in a given node
impl Salience<polarity::Supply> for UnitClass {
    fn get_value(
        self,
        _root: &Root,
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
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_unitclass_supply(self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.get_value_mult())
        }
    }
}

//this method tells us how much demand there is for a given shipclass in a given node
impl Salience<polarity::Demand> for UnitClass {
    fn get_value(
        self,
        _root: &Root,
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
            .filter(|unit| unit.get_allegiance() == faction)
            .map(|unit| unit.get_unitclass_demand(self.clone()))
            .sum::<u64>() as f32;
        if sum == 0_f32 {
            None
        } else {
            Some(sum * self.get_value_mult())
        }
    }
}

fn transpose<T>(v: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}

//TODO: make the logic apply more generally to stockpiles attached to ships

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalSalience {
    pub factionsalience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub resourcesalience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
    pub unitclasssalience: RwLock<Vec<Vec<Vec<[f32; 2]>>>>,
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
    pub shipflavors: Vec<Arc<ShipFlavor>>,
    pub squadronflavors: Vec<Arc<SquadronFlavor>>,
    pub shipclasses: Vec<Arc<ShipClass>>,
    pub squadronclasses: Vec<Arc<SquadronClass>>,
    pub shipinstances: RwLock<Vec<Arc<ShipInstance>>>,
    pub squadroninstances: RwLock<Vec<Arc<SquadronInstance>>>,
    pub unitcounter: Arc<AtomicU64>,
    pub engagements: RwLock<Vec<Arc<Engagement>>>,
    pub globalsalience: GlobalSalience,
    pub turn: Arc<AtomicU64>,
}

impl PartialEq for Root {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.nodeflavors == other.nodeflavors
            && self.nodes == other.nodes
            && self.systems == other.systems
            && self.edgeflavors == other.edgeflavors
            && self.edges == other.edges
            && self.neighbors == other.neighbors
            && self.factions == other.factions
            && self.wars == other.wars
            && self.resources == other.resources
            && self.hangarclasses == other.hangarclasses
            && self.hangarinstancecounter.load(atomic::Ordering::Relaxed)
                == other.hangarinstancecounter.load(atomic::Ordering::Relaxed)
            && self.engineclasses == other.engineclasses
            && self.repairerclasses == other.repairerclasses
            && self.factoryclasses == other.factoryclasses
            && self.shipyardclasses == other.shipyardclasses
            && self.shipais == other.shipais
            && self.shipflavors == other.shipflavors
            && self.squadronflavors == other.squadronflavors
            && self.shipclasses == other.shipclasses
            && self.squadronclasses == other.squadronclasses
            && self.shipinstances.read().unwrap().clone()
                == other.shipinstances.read().unwrap().clone()
            && self.squadroninstances.read().unwrap().clone()
                == other.squadroninstances.read().unwrap().clone()
            && self.unitcounter.load(atomic::Ordering::Relaxed)
                == other.unitcounter.load(atomic::Ordering::Relaxed)
            && self.engagements.read().unwrap().clone() == other.engagements.read().unwrap().clone()
            && self.globalsalience.factionsalience.read().unwrap().clone()
                == other.globalsalience.factionsalience.read().unwrap().clone()
            && self.globalsalience.resourcesalience.read().unwrap().clone()
                == other
                    .globalsalience
                    .resourcesalience
                    .read()
                    .unwrap()
                    .clone()
            && self
                .globalsalience
                .unitclasssalience
                .read()
                .unwrap()
                .clone()
                == other
                    .globalsalience
                    .unitclasssalience
                    .read()
                    .unwrap()
                    .clone()
            && self.turn.load(atomic::Ordering::Relaxed)
                == other.turn.load(atomic::Ordering::Relaxed)
    }
}

impl Root {
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
        location.insert_unit(new_ship.get_unit());
        new_ship
    }
    pub fn engagement_check(&self, node: Arc<Node>, actor: Arc<Faction>) -> Option<Arc<Faction>> {
        if node.mutables.read().unwrap().check_for_battles {
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
        } else {
            None
        }
    }
    pub fn internal_battle(&self, data: EngagementPrep) -> Engagement {
        let mut rng = Hc128Rng::seed_from_u64(47);

        let duration = data.calculate_engagement_duration(self, &mut rng);

        //we take the reinforcement data from our engagementprep and convert the distances to the scaling factor for travel time
        //for what percentage of the battle's duration the unit will be present
        let coalition_strengths = data.get_coalition_strengths(duration);

        let coalition_objective_difficulties = data.get_coalition_objective_difficulties();

        let coalition_chances: HashMap<u64, f32> = data
            .coalitions
            .iter()
            .map(|(index, faction_map)| {
                let chance: f32 = *coalition_strengths.get(index).unwrap() as f32
                    * *coalition_objective_difficulties.get(index).unwrap() as f32
                    * faction_map
                        .keys()
                        .map(|faction| faction.battlescalar)
                        .product::<f32>()
                    * Normal::<f32>::new(1.0, self.config.battlescalars.attacker_chance_dev)
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

        let neighbors = self.neighbors.get(&data.location).unwrap();

        let duration_damage_rand = Normal::<f32>::new(1.0, self.config.battlescalars.damage_dev)
            .unwrap()
            .sample(&mut rng)
            .clamp(0.0, 1.0);

        //NOTE: Maybe have the lethality scaling over battle duration be logarithmic? Maybe modder-specified?
        let unit_status: HashMap<u64, HashMap<Arc<Faction>, HashMap<Unit, UnitStatus>>> = data
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
                                        .map(|(_, units)| {
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
                                                if unit.get_mother_node() == data.location {
                                                    unit.get_location()
                                                } else {
                                                    UnitLocation::Node(data.location.clone())
                                                }
                                            }
                                            false => {
                                                if unit.is_in_node() {
                                                    UnitLocation::Node(
                                                        unit.navigate(self, neighbors)
                                                            .unwrap_or(data.location.clone()),
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
                                                data.wars.contains(&(
                                                    *index.min(rhs_index),
                                                    **rhs_index.max(&index),
                                                ))
                                            })
                                            .map(|(_, strength)| *strength)
                                            .sum::<u64>()
                                            as f32;
                                        let (damage, engine_damage) = unit.calculate_damage(
                                            self,
                                            is_victor,
                                            allied_strength,
                                            enemy_strength,
                                            duration,
                                            duration_damage_rand,
                                            &mut rng,
                                        );
                                        let is_alive = unit.get_hull() > damage;
                                        (
                                            unit.clone(),
                                            UnitStatus {
                                                location: match is_alive {
                                                    true => Some(new_location),
                                                    false => None,
                                                },
                                                damage,
                                                engine_damage,
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
            visiblename: format!("Battle of {}", data.location.visiblename.clone()),
            turn: data.turn,
            coalitions: data.coalitions,
            aggressor: data.aggressor.clone(),
            objectives: HashMap::new(),
            location: data.location,
            duration,
            victors: (victor, victor_coalition),
            unit_status,
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
        dead.iter()
            .for_each(|ship| match &ship.mutables.read().unwrap().location {
                UnitLocation::Node(node) => node
                    .mutables
                    .write()
                    .unwrap()
                    .units
                    .retain(|unit| unit.get_id() != ship.id),
                UnitLocation::Squadron(squadron) => squadron
                    .mutables
                    .write()
                    .unwrap()
                    .daughters
                    .retain(|unit| unit.get_id() != ship.id),
                UnitLocation::Hangar(hangar) => hangar
                    .mutables
                    .write()
                    .unwrap()
                    .contents
                    .retain(|unit| unit.get_id() != ship.id),
            });
        dead.iter().for_each(|ship| {
            ship.clone().kill();
        });
        self.shipinstances
            .write()
            .unwrap()
            .retain(|ship| ship.mutables.read().unwrap().hull > 0);
    }
    pub fn disband_squadrons(&self) {
        let dead = self
            .squadroninstances
            .read()
            .unwrap()
            .iter()
            .filter(|squadron| {
                ((squadron.get_strength(self.config.battlescalars.avg_duration) as f32)
                    < (squadron.idealstrength as f32 * squadron.class.disbandthreshold))
                    && !squadron.mutables.read().unwrap().ghost
            })
            .cloned()
            .collect::<Vec<_>>();
        for squadron in dead {
            squadron
                .get_daughters()
                .iter()
                .all(|daughter| daughter.transfer(UnitLocation::Node(squadron.get_mother_node())));
        }
        let remaining: Vec<Arc<SquadronInstance>> = self
            .squadroninstances
            .read()
            .unwrap()
            .iter()
            .filter(|squadron| {
                squadron.mutables.read().unwrap().ghost || !squadron.get_daughters().is_empty()
            })
            .cloned()
            .collect();
        self.squadroninstances
            .write()
            .unwrap()
            .retain(|squadron| remaining.contains(squadron));
    }
    //oh god
    pub fn calculate_values<S: Salience<P> + Clone, P: Polarity>(
        //we need a salience, which is the type of resource or shipclass or whatever we're calculating values for
        //and the faction for which we're calculating values
        //and we specify the number of times we want to calculate these values, (NOTE: uncertain) i.e. the number of edges we'll propagate across
        &self,
        salience: S,
        subject_faction: Arc<Faction>,
        deg_mult: f32,
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
                        &self,
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
        let tagged_threats: Vec<Vec<[f32; 2]>> =
            transpose(&self.globalsalience.factionsalience.read().unwrap()[subject_faction.id]);
        //this is the factor by which a salience passing through each node should be multiplied
        //we sum the tagged threats for each node -- which are valenced according to relations with the subject faction
        //then we use Alyssa's black mathemagics to convert them so that the scaling curve is correct
        //Length equals all nodes
        //This is a subjective map for subject faction
        let node_degradations: Vec<f32> = tagged_threats
            .iter()
            .map(|factions_vec| {
                let sum = factions_vec.iter().map(|[supply, _]| supply).sum();
                scale_from_threat(sum, 20_f32) * deg_mult * 0.8
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
                self.edges.iter().filter(|(_, flavor)| flavor.propagates).for_each(|((a, b), _)| {
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
                        if objectfaction.propagates {
                            let supply = self.calculate_values::<Arc<Faction>, polarity::Supply>(
                                objectfaction.clone(),
                                subjectfaction.clone(),
                                self.config.saliencescalars.faction_deg_mult,
                                self.config.saliencescalars.faction_prop_iters,
                            );
                            let demand = self.calculate_values::<Arc<Faction>, polarity::Demand>(
                                objectfaction.clone(),
                                subjectfaction.clone(),
                                self.config.saliencescalars.faction_deg_mult,
                                self.config.saliencescalars.faction_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
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
                        if resource.propagates {
                            let supply = self.calculate_values::<Arc<Resource>, polarity::Supply>(
                                resource.clone(),
                                faction.clone(),
                                self.config.saliencescalars.resource_deg_mult,
                                self.config.saliencescalars.resource_prop_iters,
                            );
                            let demand = self.calculate_values::<Arc<Resource>, polarity::Demand>(
                                resource.clone(),
                                faction.clone(),
                                self.config.saliencescalars.resource_deg_mult,
                                self.config.saliencescalars.resource_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    })
                    .collect()
            })
            .collect()
    }
    pub fn calculate_global_unitclass_salience(&self) -> Vec<Vec<Vec<[f32; 2]>>> {
        self.factions
            .par_iter()
            .map(|faction| {
                self.shipclasses
                    .par_iter()
                    .map(|shipclass| {
                        if shipclass.propagates {
                            let supply = self.calculate_values::<UnitClass, polarity::Supply>(
                                ShipClass::get_unitclass(shipclass.clone()),
                                faction.clone(),
                                self.config.saliencescalars.unitclass_deg_mult,
                                self.config.saliencescalars.unitclass_prop_iters,
                            );
                            let demand = self.calculate_values::<UnitClass, polarity::Demand>(
                                ShipClass::get_unitclass(shipclass.clone()),
                                faction.clone(),
                                self.config.saliencescalars.unitclass_deg_mult,
                                self.config.saliencescalars.unitclass_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    })
                    .chain(self.squadronclasses.par_iter().map(|squadronclass| {
                        if squadronclass.propagates {
                            let supply = self.calculate_values::<UnitClass, polarity::Supply>(
                                SquadronClass::get_unitclass(squadronclass.clone()),
                                faction.clone(),
                                self.config.saliencescalars.unitclass_deg_mult,
                                self.config.saliencescalars.unitclass_prop_iters,
                            );
                            let demand = self.calculate_values::<UnitClass, polarity::Demand>(
                                SquadronClass::get_unitclass(squadronclass.clone()),
                                faction.clone(),
                                self.config.saliencescalars.unitclass_deg_mult,
                                self.config.saliencescalars.unitclass_prop_iters,
                            );
                            supply
                                .iter()
                                .zip(demand.iter())
                                .map(|(s, d)| [*s, *d])
                                .collect()
                        } else {
                            self.nodes.iter().map(|_| [0.0; 2]).collect()
                        }
                    }))
                    .collect()
            })
            .collect()
    }
    pub fn process_turn(&mut self) {
        let turn_start = Instant::now();
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
        //That's what'll happen here with the way it is currently. Previous turn's values get used in calc faction salience, then get overwritten.
        //At least assuming the order of operations works like I'm expecting -- check that with Amaryllis.
        //On turn 1, we'll want to run calc faction salience several times to get the values to settle.
        let salience_propagation_start = Instant::now();
        let gfs = self.calculate_global_faction_salience();
        *self.globalsalience.factionsalience.write().unwrap() = gfs;
        let grs = self.calculate_global_resource_salience();
        *self.globalsalience.resourcesalience.write().unwrap() = grs;
        let gus = self.calculate_global_unitclass_salience();
        *self.globalsalience.unitclasssalience.write().unwrap() = gus;
        let salience_propagation_finished = salience_propagation_start.elapsed();
        dbg!(salience_propagation_finished);

        //run operation management logic

        //move ships, one edge at a time
        //running battle checks and stockpile balancing with each traversal
        let ship_moves_start = Instant::now();
        let shipinstances = self.shipinstances.read().unwrap().clone();
        shipinstances.iter().for_each(|shipinstance| {
            let mut moving = true;
            while moving {
                if (!shipinstance.is_in_node()) || shipinstance.maneuver(&self).is_none() {
                    moving = false
                }
            }
        });
        dbg!(ship_moves_start.elapsed());

        //move squadrons, one edge at a time
        //running battle checks and stockpile balancing with each traversal

        //run defection logic

        //run diplomacy logic

        //transmit root data to frontend

        let number_of_ships = &self.shipinstances.read().unwrap().len();

        dbg!(number_of_ships);

        dbg!(turn_start.elapsed());
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
    #[test]
    fn nav_calcs_test() {
        use ordered_float::NotNan;
        use rand::prelude::*;
        let mut rng = rand_hc::Hc128Rng::seed_from_u64(5803495084);
        let mut rdavs = Vec::new();
        let mut rsavs = Vec::new();
        for _ in 0..50 {
            let supply = rng.gen_range(0.0..10.0);
            let demand = rng.gen_range(0.0..10.0);
            let rdav = (demand - supply);
            let rsav = ((supply * demand) + ((supply - demand) * 5.0)) / 10.0;
            rdavs.push(NotNan::new(rdav).unwrap());
            rsavs.push(NotNan::new(rsav).unwrap());
            println!("Supply: {}; demand: {}", supply, demand);
            println!("Demand attraction value: {}", rdav);
            println!("Supply attraction value: {}", rsav);
            println!();
            println!();
        }
        println!(
            "Demand attraction min: {}; max: {}",
            rdavs.iter().min().unwrap(),
            rdavs.iter().max().unwrap()
        );
        println!(
            "Demand attraction average: {}",
            rdavs.iter().sum::<NotNan<f32>>() / rdavs.len() as f32
        );
        println!(
            "Supply attraction min: {}; max: {}",
            rsavs.iter().min().unwrap(),
            rsavs.iter().max().unwrap()
        );
        println!(
            "Supply attraction average: {}",
            rsavs.iter().sum::<NotNan<f32>>() / rsavs.len() as f32
        );
    }
}
