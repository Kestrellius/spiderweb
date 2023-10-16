use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer_pretty};
use spiderweb::{internal, json};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::time::{Duration, Instant};

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
    let empire = root
        .factions
        .get(internal::Key::<internal::Faction>::new_from_index(0));
    let rebels = root
        .factions
        .get(internal::Key::<internal::Faction>::new_from_index(1));
    let pirates = root
        .factions
        .get(internal::Key::<internal::Faction>::new_from_index(2));
    let steel = root
        .resources
        .get(internal::Key::<internal::Resource>::new_from_index(0));
    let components = root
        .resources
        .get(internal::Key::<internal::Resource>::new_from_index(1));

    while root.turn < 5 {
        root.process_turn();
    }

    for i in 0..10 {
        root.update_node_threats(10);
        //dbg!();
    }

    //dbg!(&root.nodes);

    root.process_turn();

    /*root.nodes.iter().for_each(|node| {
        let mut threat_list: Vec<(internal::Key<internal::Faction>, f32)> =
            node.threat.iter().map(|(fid, v)| (*fid, *v)).collect();
        threat_list.sort_by_key(|(id, _)| *id);
        println!("{}", node.visiblename);
        for (_, threat) in threat_list {
            print!("{:.6}, ", threat);
        }
        print!("\n");
    })*/

    let salience_values = root
        .calculate_values::<internal::Key<internal::Resource>, internal::polarity::Supply>(
            internal::Key::<internal::Resource>::new_from_index(1),
            internal::Key::<internal::Faction>::new_from_index(0),
            5,
        );
    salience_values
        .iter()
        .copied()
        .zip(root.nodes.iter())
        .for_each(|(value, (_, node))| println!("{:.3}\t{}", value, node.visiblename));
}
