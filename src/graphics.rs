use std::f64::consts::PI;

use nalgebra::Vector2;
use plotly::common::{Line, Marker, Mode};
use plotly::layout::themes::PLOTLY_WHITE;
use plotly::layout::Axis;
use plotly::plot::Plot;

use plotly::{Layout, Scatter};

use crate::structs::Halfspace;

// type Arrow = Box<Scatter<(f64, f64), (f64, f64)>>;

fn create_arrow(
    start: Vector2<f64>,
    end: Vector2<f64>,
    // arrow_size: f64,
    // color: &str,
) -> Box<Scatter<f64, f64>> {
    // let direction = (end - start).normalize();
    // let angle = PI / 6.0; // 30 degrees
    // let perpendicular = Vector2::new(-direction[1], direction[0]).normalize();

    // let arrow_point1 = end + arrow_size * (direction * angle.cos() + perpendicular * angle.sin());
    // let arrow_point2 = end + arrow_size * (direction * angle.cos() - perpendicular * angle.sin());

    Scatter::new(vec![start[0], end[0]], vec![start[1], end[1]])
        .name("osv")
        .mode(Mode::Lines)
        .line(Line::new().color("green"))

    // let arrowhead = Scatter::new(
    //     vec![end[0], arrow_point1[0], arrow_point2[0], end[0]], // Close the arrowhead
    //     vec![end[1], arrow_point1[1], arrow_point2[1], end[1]],
    // )
    // .mode(Mode::Lines)
    // .line(Line::new().color(color));
}

pub fn visualize2d(boundary: &[Halfspace<2>]) {
    let mut plot = Plot::new();
    let trace = Scatter::new(
        boundary.iter().map(|hs| hs.b[0]).collect::<Vec<f64>>(),
        boundary.iter().map(|hs| hs.b[1]).collect::<Vec<f64>>(),
    )
    .name("boundary")
    .mode(Mode::Markers);
    plot.add_trace(trace);

    for hs in boundary {
        let start = *hs.b;
        let end = start + hs.n;
        plot.add_trace(create_arrow(start, end));
    }

    plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().range(vec![0.0, 1.0]))
            .y_axis(Axis::new().range(vec![0.0, 1.0])),
    );

    plot.show();
}
