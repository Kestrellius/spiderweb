use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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
    pub visible_name: String, //faction name as shown to player
    pub description: String,
    pub visibility: bool,
    pub propagates: bool,
    pub efficiency_default: f32, //starting value for production facility efficiency
    pub efficiency_target: f32, //end value for efficiency, toward which efficiency changes over time in a node held by this faction
    pub efficiency_delta: f32,  //rate at which efficiency changes
    pub battle_scalar: f32,
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
