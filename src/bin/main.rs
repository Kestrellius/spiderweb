use spiderweb::connection;
use spiderweb::hydration;
use std::fs::File;
use std::time::Instant;

fn main() {
    //rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    //env::set_var("RUST_BACKTRACE", "1");
    let file = File::open("benchmark.json").unwrap();
    let start = Instant::now();
    let json_root: hydration::Root = serde_json::from_reader(file).unwrap();
    let duration = start.elapsed();
    dbg!(duration);
    let mut root_0 = json_root.hydrate();

    for _i in 0..5 {
        root_0.process_turn();
    }
    let connection_root_0 = connection::Root::desiccate(&root_0);

    let mut root_1 = &mut connection_root_0.rehydrate();

    let connection_root_1 = connection::Root::desiccate(root_1);

    let mut root_2 = connection::Root::rehydrate(connection_root_1.clone());

    dbg!(*root_1 == root_2);

    let connection_root_2 = connection::Root::desiccate(&root_2);

    dbg!(&connection_root_1 == &connection_root_2);

    let mut root_3 = connection::Root::rehydrate(connection_root_2);

    for _i in 0..20 {
        root_3.process_turn();
    }

    //let json_connection_root = serde_json::to_string_pretty(&connection_root_1).unwrap();

    //std::fs::write(&"connection_root.json", &json_connection_root);

    dbg!(root_3.nodes.len());
    dbg!(root_3.edges.len());

    root_3
        .squadrons
        .read()
        .unwrap()
        .iter()
        .for_each(|squadron| {
            println!("Squadron {}:", squadron.visible_name);
            println!("Ghost: {}", squadron.mutables.read().unwrap().ghost);
            dbg!(squadron.unit_container.read().unwrap().contents.len());
        });

    /*
    let nodes_w_shipyards = root
        .nodes
        .iter()
        .filter(|node| node.mutables.read().unwrap().shipyardinstancelist.len() > 0)
        .map(|node| node.visiblename.clone())
        .collect::<Vec<_>>();

    dbg!(nodes_w_shipyards);
    */

    /*
    dbg!(root.nodes);
    dbg!(root.shipinstances);
    dbg!(root.shipinstancecounter);

    /*
    let print = json_hydration::generate_saliencenames();

    let mut paste: ClipboardContext = ClipboardProvider::new().unwrap();
    paste.set_contents(print).unwrap();
    */

    let faction_a = internal::Key::<internal::Faction>::new_from_index(0);
    let steel = internal::Key::<internal::Resource>::new_from_index(0);
    let components = internal::Key::<internal::Resource>::new_from_index(1);
    let food = internal::Key::<internal::Resource>::new_from_index(2);
    let personnel = internal::Key::<internal::Resource>::new_from_index(3);
    let fighter_a = internal::Key::<internal::ShipClass>::new_from_index(1);
    let fighter_b = internal::Key::<internal::ShipClass>::new_from_index(2);
    let cap_a = internal::Key::<internal::ShipClass>::new_from_index(3);
    let cap_b = internal::Key::<internal::ShipClass>::new_from_index(4);
    */

    /*
    while root.turn < 2 {
        root.process_turn();
    }

    for _ in 0..10 {
        root.update_node_threats(10);
        //dbg!();
    }
    */

    //dbg!(&root.nodes);

    //root.process_turn();

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

    /*
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

    root.shipinstances.iter().for_each(|x| {
        let class = &root.shipclasses.get(x.1.shipclass).unwrap().visiblename;
        let allegiance = &root.factions.get(x.1.allegiance).unwrap().visiblename;
        let location = &root
            .nodes
            .get({
                match x.1.location {
                    ShipLocationFlavor::Node(k) => k,
                    _ => internal::Key::<internal::Node>::new_from_index(0),
                }
            })
            .unwrap()
            .visiblename;
        println!(
            "Name: {:?}
        Class: {:?}
        Allegiance: {:?}
        Location: {:?}",
            x.1.visiblename, class, allegiance, location
        )
    });
    let mut stockpile1 = PluripotentStockpile {
        visibility: false,
        resource_contents: HashMap::from([(steel, 0), (components, 0)]),
        ship_contents: HashSet::from([
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
        ]),
        allowed: Some((Vec::new(), vec![fighter_a])),
        target: 500,
        capacity: 1000,
        propagate: false,
    };
    let mut stockpile2 = PluripotentStockpile {
        visibility: false,
        resource_contents: HashMap::from([
            (steel, 150),
            (components, 100),
            (food, 200),
            (personnel, 50),
        ]),
        ship_contents: HashSet::from([
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_a,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
            root.create_ship(
                fighter_b,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                faction_a,
            ),
        ]),
        allowed: Some((
            vec![steel, components, food, personnel],
            vec![fighter_a, fighter_b, cap_a, cap_b],
        )),
        target: 500,
        capacity: 1000000,
        propagate: false,
    };
    println!("{:#?}", stockpile1.collate_contents(&root));
    println!("{:#?}", stockpile2.collate_contents(&root));

    let quantity = 5;

    println!(
        "Attempting to transfer {} fighter_bs from Stockpile2 to Stockpile1.",
        quantity
    );
    stockpile2.transfer(
        &mut stockpile1,
        &root,
        CollatedCargo::ShipClass(fighter_b),
        quantity,
    );

    println!("{:#?}", stockpile1.collate_contents(&root));
    println!("{:#?}", stockpile2.collate_contents(&root));
    root.edges.iter().for_each(|((a, b), _)| {
        println!(
            "{}, {}",
            root.nodes.get(*a).unwrap().visiblename,
            root.nodes.get(*b).unwrap().visiblename
        )
    });
    */
}
