use spiderweb::json;
use std::fs::File;
use std::time::Instant;

fn main() {
    //env::set_var("RUST_BACKTRACE", "1");
    let file = File::open("mod-specs.json").unwrap();
    let start = Instant::now();
    let json_root: json::Root = serde_json::from_reader(file).unwrap();
    let duration = start.elapsed();
    dbg!(duration);
    let mut root = json_root.hydrate();

    for _i in 0..50 {
        root.process_turn();
    }

    dbg!(root.nodes.len());
    dbg!(root.resources.len());
    dbg!(root.shipclasses.len());

    /*
        dbg!(root.nodes);
        dbg!(root.shipinstances);
        dbg!(root.shipinstancecounter);

    let empire = internal::Key::<internal::Faction>::new_from_index(0);
    let steel = internal::Key::<internal::Resource>::new_from_index(0);
    let components = internal::Key::<internal::Resource>::new_from_index(1);
    let food = internal::Key::<internal::Resource>::new_from_index(2);
    let personnel = internal::Key::<internal::Resource>::new_from_index(3);
    let tieln = internal::Key::<internal::ShipClass>::new_from_index(1);
    let z95 = internal::Key::<internal::ShipClass>::new_from_index(2);
    let isd = internal::Key::<internal::ShipClass>::new_from_index(3);
    let nebulonb = internal::Key::<internal::ShipClass>::new_from_index(4);
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
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
        ]),
        allowed: Some((Vec::new(), vec![tieln])),
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
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                tieln,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
            root.create_ship(
                z95,
                ShipLocationFlavor::Node(internal::Key::<internal::Node>::new_from_index(0)),
                empire,
            ),
        ]),
        allowed: Some((
            vec![steel, components, food, personnel],
            vec![tieln, z95, isd, nebulonb],
        )),
        target: 500,
        capacity: 1000000,
        propagate: false,
    };
    println!("{:#?}", stockpile1.collate_contents(&root));
    println!("{:#?}", stockpile2.collate_contents(&root));

    let quantity = 5;

    println!(
        "Attempting to transfer {} Z95s from Stockpile2 to Stockpile1.",
        quantity
    );
    stockpile2.transfer(
        &mut stockpile1,
        &root,
        CollatedCargo::ShipClass(z95),
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
