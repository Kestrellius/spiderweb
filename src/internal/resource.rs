use crate::internal::engagement::UnitStatus;
use crate::internal::faction::{Faction, FactionID};
use crate::internal::node::{EdgeFlavor, Locality, Node, NodeFlavor};
use crate::internal::root::Root;
use crate::internal::unit::{Mobility, Ship, ShipClass, ShipClassID, ShipFlavor, Unit};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::iter;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub id: usize,
    pub visible_name: String,
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
    fn get_pluripotent_transaction_resource_demand(
        &self,
        resource: Arc<Resource>,
        external_demand: f32,
        speed_factor: f32,
        total_target: f32,
    ) -> u64;
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
    fn get_pluripotent_transaction_resource_demand(
        &self,
        resource: Arc<Resource>,
        external_demand: f32,
        speed_factor: f32,
        total_target: f32,
    ) -> u64 {
        self.0.get_pluripotent_transaction_resource_demand(
            resource.clone(),
            external_demand,
            speed_factor,
            total_target,
        ) + self.1.get_pluripotent_transaction_resource_demand(
            resource.clone(),
            external_demand,
            speed_factor,
            total_target,
        )
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
    pub resource_type: Arc<Resource>,
    pub contents: u64,
    pub rate: u64,
    pub target: u64,
    pub capacity: u64,
    pub propagates: bool,
}

impl UnipotentStockpile {
    pub fn input_is_sufficient(&self) -> bool {
        self.contents >= self.rate
    }
    //this is the logic to determine whether a unipotent stockpile should be active, dormant, or stalled
    pub fn output_state(&self) -> OutputState {
        //NOTE: Dormancy is dummied out for now
        //if self.contents >= self.target {
        //    OutputState::Dormant
        //} else
        //NOTE: This is not perfect because we don't have easy access to efficiency here,
        //so we don't know exactly how much will be added to the stockpile the next time it's incremented.
        //However, if capacity is exceeded, the contents will be capped out properly in output_process,
        //and the next time output_state is run, it will detect as stalled.
        if self.contents + self.rate >= self.capacity {
            OutputState::Stalled
        } else {
            OutputState::Active
        }
    }
    pub fn input_process(&mut self) {
        let subtracted: Option<u64> = self.contents.checked_sub(self.rate);
        if let Some(new) = subtracted {
            self.contents = new;
        } else {
            panic!("Factory input stockpile is too low.")
        }
    }
    pub fn output_process(&mut self, efficiency: f32) {
        self.contents += (self.rate as f32 * efficiency) as u64;
        if self.contents >= self.capacity {
            self.contents = self.capacity
        }
    }
}

impl Stockpileness for UnipotentStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((self.resource_type.clone(), self.contents)).collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resource_type {
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
        Some(vec![self.resource_type.clone()])
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resource_type {
            (self.contents).saturating_sub(self.target)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resource_type {
            self.target.saturating_sub(self.contents)
        } else {
            0
        }
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        _resource: Arc<Resource>,
        _external_demand: f32,
        _speed_factor: f32,
        _total_target: f32,
    ) -> u64 {
        0
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resource_type {
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
        if cargo == self.resource_type {
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
        let sum = self.get_resource_num(resource.clone());
        sum
    }
    fn get_resource_demand(&self, resource: Arc<Resource>) -> u64 {
        if self
            .allowed
            .as_ref()
            .map(|allowed| allowed.contains(&resource))
            .unwrap_or(true)
        {
            self.target.saturating_sub(self.get_fullness())
        } else {
            0
        }
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        resource: Arc<Resource>,
        external_demand: f32,
        speed_factor: f32,
        total_naive_demand: f32,
    ) -> u64 {
        if self
            .allowed
            .as_ref()
            .map(|allowed| allowed.contains(&resource))
            .unwrap_or(true)
        {
            let space_left_without_resource = self.target.saturating_sub(
                self.get_fullness()
                    .saturating_sub(self.get_resource_num(resource.clone())),
            );
            ((external_demand
                * speed_factor
                * (space_left_without_resource as f32 / total_naive_demand)) as u64)
                .clamp(0, space_left_without_resource) as u64
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
    pub resource_type: Arc<Resource>,
    pub contents: Arc<AtomicU64>,
    pub rate: u64,
    pub capacity: u64,
}

impl Stockpileness for SharedStockpile {
    fn collate_contents(&self) -> HashMap<Arc<Resource>, u64> {
        iter::once((
            self.resource_type.clone(),
            self.contents.load(atomic::Ordering::SeqCst),
        ))
        .collect()
    }
    fn get_resource_num(&self, cargo: Arc<Resource>) -> u64 {
        if cargo == self.resource_type {
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
        Some(vec![self.resource_type.clone()])
    }
    fn get_resource_supply(&self, resource: Arc<Resource>) -> u64 {
        if resource == self.resource_type {
            self.contents.load(atomic::Ordering::SeqCst)
        } else {
            0
        }
    }
    fn get_resource_demand(&self, _resourceid: Arc<Resource>) -> u64 {
        0
    }
    fn get_pluripotent_transaction_resource_demand(
        &self,
        _resource: Arc<Resource>,
        _external_demand: f32,
        _speed_factor: f32,
        _total_target: f32,
    ) -> u64 {
        0
    }
    fn insert(&mut self, cargo: Arc<Resource>, quantity: u64) -> u64 {
        if cargo == self.resource_type {
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
        if cargo == self.resource_type {
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

pub trait ResourceProcess {
    fn get_state(&self) -> FactoryState;
    fn get_resource_supply_total(&self, resource: Arc<Resource>) -> u64;
    fn get_resource_demand_total(&self, resource: Arc<Resource>) -> u64;
    fn get_target_total(&self) -> u64;
}

#[derive(Debug, Clone)]
pub struct EngineClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub base_health: Option<u64>,
    pub toughness_scalar: f32,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub speed: u64, //the number of edges this engine will allow a ship to traverse per turn, followed by the number of turns it must wait before moving again
    pub cooldown: u64,
}

impl EngineClass {
    pub fn instantiate(class: Arc<Self>) -> Engine {
        Engine {
            class: class.clone(),
            visibility: class.visibility,
            health: class.base_health,
            inputs: class.inputs.clone(),
            forbidden_nodeflavors: class.forbidden_nodeflavors.clone(),
            forbidden_edgeflavors: class.forbidden_edgeflavors.clone(),
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
pub struct Engine {
    pub class: Arc<EngineClass>,
    pub visibility: bool,
    pub health: Option<u64>,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the engine won't allow a ship to enter nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the engine won't allow a ship to traverse edges of these flavors
    pub last_move_turn: u64,
}

impl Engine {
    pub fn check_engine(
        &self,
        root: &Root,
        location: Arc<Node>,
        destinations: &Vec<Arc<Node>>,
    ) -> Option<(Vec<Arc<Node>>, u64)> {
        if (self.health != Some(0))
            && ((root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn)
                > self.class.cooldown)
            && (self.get_state() == FactoryState::Active)
        {
            let viable: Vec<Arc<Node>> = destinations
                .iter()
                .filter(|destination| {
                    self.nav_check(root, location.clone(), destination.clone().clone())
                })
                .cloned()
                .collect();
            if !viable.is_empty() {
                Some((viable, self.class.speed))
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn check_engine_movement_only(&self, turn: u64) -> bool {
        if (self.health != Some(0))
            && ((turn - self.last_move_turn) > self.class.cooldown)
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
    pub fn process_engine(
        &mut self,
        root: &Root,
        location: Arc<Node>,
        destination: Arc<Node>,
    ) -> Option<u64> {
        if (self.health != Some(0))
            && ((root.turn.load(atomic::Ordering::Relaxed) - self.last_move_turn)
                > self.class.cooldown)
            && (self.get_state() == FactoryState::Active)
            && (self.nav_check(root, location, destination))
        {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.last_move_turn = root.turn.load(atomic::Ordering::Relaxed);
            Some(self.class.speed)
        } else {
            None
        }
    }
    pub fn nav_check(&self, root: &Root, location: Arc<Node>, destination: Arc<Node>) -> bool {
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
    pub fn get_moves_left(&self, movement_left: u64) -> u64 {
        (self
            .inputs
            .iter()
            .map(|sp| sp.contents / sp.rate)
            .min()
            .unwrap_or(0))
        .min(movement_left / self.class.speed)
    }
}

impl ResourceProcess for Engine {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
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
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub repair_points: i64,
    pub repair_factor: f32,
    pub engine_repair_points: i64,
    pub engine_repair_factor: f32,
    pub subsystem_repair_points: i64,
    pub subsystem_repair_factor: f32,
    pub per_engagement: bool, //whether this repairer is run once per turn, or after every engagement
}

impl RepairerClass {
    pub fn instantiate(class: Arc<Self>) -> Repairer {
        Repairer {
            class: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
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
pub struct Repairer {
    pub class: Arc<RepairerClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
}

impl Repairer {
    pub fn process(&mut self) {
        self.inputs
            .iter_mut()
            .for_each(|input| input.input_process());
    }
}

impl ResourceProcess for Repairer {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
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
pub struct StrategicWeaponClass {
    pub id: usize,
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub forbidden_nodeflavors: Vec<Arc<NodeFlavor>>, //the weapon won't fire into nodes of these flavors
    pub forbidden_edgeflavors: Vec<Arc<EdgeFlavor>>, //the weapon won't fire across edges of these flavors
    pub damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage done by a single shot, first by points and then by factor
    pub engine_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to engine done by a single shot, first by points and then by factor
    pub strategic_weapon_damage: ((i64, i64), (f32, f32)), //lower and upper bounds for damage to strategic weapon done by a single shot, first by points and then by factor
    pub accuracy: f32, //divided by target's strategicweaponevasionscalar to get hit probability as a fraction of 1.0
    pub range: u64,    //how many edges away the weapon can reach
    pub shots: (u64, u64), //lower and upper bounds for maximum number of shots the weapon fires each time it's activated
    pub targets_enemies: bool,
    pub targets_allies: bool,
    pub targets_neutrals: bool,
    pub target_relations_lower_bound: Option<f32>,
    pub target_relations_upper_bound: Option<f32>,
    pub target_priorities_class: HashMap<ShipClassID, f32>, //how strongly weapon will prioritize ships of each class; classes absent from list will default to 1.0
    pub target_priorities_flavor: HashMap<Arc<ShipFlavor>, f32>, //how strongly weapon will prioritize ships of each flavor; flavors absent from list will default to 1.0
}

impl StrategicWeaponClass {
    pub fn instantiate(class: Arc<Self>) -> StrategicWeapon {
        StrategicWeapon {
            class: class.clone(),
            visibility: class.visibility,
            inputs: class.inputs.clone(),
        }
    }
}

impl Eq for StrategicWeaponClass {}

impl Ord for StrategicWeaponClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for StrategicWeaponClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Hash for StrategicWeaponClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategicWeapon {
    pub class: Arc<StrategicWeaponClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
}

impl StrategicWeapon {
    pub fn targets_faction(
        &self,
        root: &Root,
        allegiance: &Arc<Faction>,
        faction: &Arc<Faction>,
    ) -> bool {
        let enemies = root
            .factions
            .iter()
            .cloned()
            .filter(|rhs_faction| {
                root.wars.contains(&(
                    (rhs_faction).min(&allegiance).clone(),
                    (allegiance).max(&rhs_faction).clone(),
                ))
            })
            .collect::<Vec<_>>();
        (self.class.targets_enemies && enemies.contains(&faction))
            || (self.class.targets_allies && faction == allegiance)
            || (self.class.targets_neutrals
                && !(faction == allegiance || enemies.contains(&faction)))
            || (self
                .class
                .target_relations_upper_bound
                .map(|val| {
                    allegiance
                        .relations
                        .get(&FactionID::new_from_index(faction.id))
                        .unwrap()
                        < &val
                })
                .unwrap_or(false))
            || (self
                .class
                .target_relations_upper_bound
                .map(|val| {
                    allegiance
                        .relations
                        .get(&FactionID::new_from_index(faction.id))
                        .unwrap()
                        > &val
                })
                .unwrap_or(false))
    }
    pub fn fire<R: Rng>(
        &mut self,
        root: &Root,
        mother: Arc<Ship>,
        mut rng: &mut R,
    ) -> HashMap<Unit, UnitStatus> {
        let allegiance = &mother.get_allegiance();
        let location = mother.get_mother_node();

        let target_nodes = location.get_nodes_in_range(
            root,
            self.class.range,
            &self.class.forbidden_nodeflavors,
            &self.class.forbidden_edgeflavors,
        );

        let target_factions = root
            .factions
            .iter()
            .cloned()
            .filter(|faction| self.targets_faction(root, &allegiance, faction))
            .collect::<Vec<_>>();

        let targets: Vec<(Arc<Ship>, f32)> = target_nodes
            .iter()
            .map(|node| {
                node.unit_container
                    .read()
                    .unwrap()
                    .contents
                    .iter()
                    .map(|unit| unit.get_undocked_daughters())
                    .flatten()
                    .filter(|ship| ship.is_alive())
                    .filter(|ship| {
                        target_factions.contains(&&ship.mutables.read().unwrap().allegiance)
                            && ship.id != mother.id
                    })
                    .map(|ship| {
                        (
                            ship.clone(),
                            self.class
                                .target_priorities_class
                                .get(&ShipClassID::new_from_index(ship.class.id))
                                .unwrap_or(&0.0)
                                + self
                                    .class
                                    .target_priorities_flavor
                                    .get(&ship.class.shipflavor)
                                    .unwrap_or(&0.0),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        let shots_fired: usize = rng.gen_range(self.class.shots.0..self.class.shots.1) as usize;

        (0..shots_fired).for_each(|_| self.inputs.iter_mut().for_each(|sp| sp.input_process()));

        let hit_ships: Vec<Arc<Ship>> = (0..shots_fired)
            .map(|_| {
                targets
                    .choose_weighted(&mut rng, |(_, weight)| weight.clone())
                    .unwrap()
            })
            .map(|(ship, _)| ship.clone())
            .filter(|target| {
                (self.class.accuracy / target.class.strategic_weapon_evasion_scalar) > 1.0
            })
            .collect();

        let hit_ships_status: HashMap<Unit, UnitStatus> = hit_ships
            .iter()
            .map(|hit_ship| {
                let status = UnitStatus {
                    location: Some(hit_ship.mutables.read().unwrap().location.clone()),
                    damage: ((rng
                        .gen_range(self.class.damage.0 .0 as f32..self.class.damage.0 .1 as f32)
                        / hit_ship.class.toughness_scalar)
                        + ((rng.gen_range(
                            self.class.damage.1 .0 as f32..self.class.damage.1 .1 as f32,
                        ) / hit_ship.class.toughness_scalar)
                            * hit_ship.class.base_hull as f32)) as i64,
                    engine_damage: hit_ship
                        .mutables
                        .read()
                        .unwrap()
                        .engines
                        .iter()
                        .filter(|e| e.health.is_some())
                        .map(|e| {
                            ((rng.gen_range(
                                self.class.engine_damage.0 .0 as f32
                                    ..self.class.engine_damage.0 .1 as f32,
                            ) / e.class.toughness_scalar)
                                + ((rng.gen_range(
                                    self.class.engine_damage.1 .0 as f32
                                        ..self.class.engine_damage.1 .1 as f32,
                                ) / hit_ship.class.toughness_scalar)
                                    * e.class.base_health.unwrap() as f32))
                                as i64
                        })
                        .collect(),
                    subsystem_damage: hit_ship
                        .mutables
                        .read()
                        .unwrap()
                        .subsystems
                        .iter()
                        .filter(|s| s.health.is_some())
                        .map(|s| {
                            ((rng.gen_range(
                                self.class.strategic_weapon_damage.0 .0 as f32
                                    ..self.class.strategic_weapon_damage.0 .1 as f32,
                            ) / s.class.toughness_scalar)
                                + ((rng.gen_range(
                                    self.class.strategic_weapon_damage.1 .0 as f32
                                        ..self.class.strategic_weapon_damage.1 .1 as f32,
                                ) / hit_ship.class.toughness_scalar)
                                    * s.class.base_health.unwrap() as f32))
                                as i64
                        })
                        .collect(),
                };
                (hit_ship.get_unit(), status)
            })
            .collect();
        hit_ships_status.iter().for_each(|(unit, status)| {
            unit.damage(
                status.damage,
                &status.engine_damage,
                &status.subsystem_damage,
            )
        });
        root.remove_dead();
        hit_ships_status
    }
}

impl ResourceProcess for StrategicWeapon {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
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
    pub visible_name: String,
    pub description: String,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl FactoryClass {
    pub fn instantiate(class: Arc<Self>) -> Factory {
        Factory {
            class: class.clone(),
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
pub struct Factory {
    //this is an actual factory, derived from a factory class
    pub class: Arc<FactoryClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>, //the data for the factory's asset consumption
    pub outputs: Vec<UnipotentStockpile>, //the data for the factory's asset production
}

impl Factory {
    //we take an active factory and update all its inputs and outputs to add or remove resources
    pub fn process(&mut self, location_efficiency: f32) {
        if let FactoryState::Active = self.get_state() {
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

impl ResourceProcess for Factory {
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
                let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
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
        let sum = self
            .outputs
            .iter()
            .filter(|sp| sp.propagates)
            .map(|sp| sp.get_resource_supply(resource.clone()))
            .sum::<u64>();
        sum
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
    pub visible_name: Option<String>,
    pub description: Option<String>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<ShipClassID, u64>,
    pub construct_rate: u64,
    pub efficiency: f32,
}

impl ShipyardClass {
    pub fn instantiate(class: Arc<Self>, shipclasses: &Vec<Arc<ShipClass>>) -> Shipyard {
        Shipyard {
            class: class.clone(),
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
            construct_points: 0,
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
pub struct Shipyard {
    pub class: Arc<ShipyardClass>,
    pub visibility: bool,
    pub inputs: Vec<UnipotentStockpile>,
    pub outputs: HashMap<Arc<ShipClass>, u64>,
    pub construct_points: u64,
    pub efficiency: f32,
}

impl Shipyard {
    pub fn process(&mut self, location_efficiency: f32) {
        if let FactoryState::Active = self.get_state() {
            self.inputs
                .iter_mut()
                .for_each(|stockpile| stockpile.input_process());
            self.construct_points +=
                (self.class.construct_rate as f32 * location_efficiency) as u64;
        }
    }

    pub fn try_choose_ship(
        &mut self,
        _shipclasses: &Vec<Arc<ShipClass>>,
    ) -> Option<Arc<ShipClass>> {
        //we go through the list of ships the shipyard can produce, specified as its outputs, and grab the one with the highest desirability weight
        let shipclass = self
            .outputs
            .iter()
            .max_by_key(|(_, weight)| *weight)
            .unwrap()
            .0;
        let cost = shipclass.base_strength;
        //then, if the shipyard has enough points to build it, we subtract the cost and return the ship class id
        if self.construct_points >= cost {
            self.construct_points -= cost;
            Some(shipclass.clone())
        //otherwise we return nothing
        } else {
            None
        }
    }

    //this uses try_choose_ship to generate the list of ships the shipyard is building this turn
    pub fn plan_ships(
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

impl ResourceProcess for Shipyard {
    fn get_state(&self) -> FactoryState {
        let input_is_good: bool = self.inputs.iter().all(|x| x.input_is_sufficient()); //this uses the FactoryOutput sufficiency method defined earlier
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
