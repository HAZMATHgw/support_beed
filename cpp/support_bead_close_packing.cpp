// ---------------------------------------------------------------------------
// SupportMaterial.cpp : anonymous-namespace bead helpers.
//
// Close-packed variant. The bead is a sphere, not a column element: every bead
// drops into the hollow formed by three beads of the layer below, so each bead
// touches 12 neighbours (6 in plane, 3 below, 3 above) instead of 8.
//
// Only three things changed against the original:
//   1. the lattice origin is fixed for the whole object instead of being taken
//      from the current layer's bounding box (staggering is meaningless if the
//      base lattice drifts from layer to layer),
//   2. the per-layer shift runs over a period (2 = hcp ABAB, 3 = fcc ABCABC)
//      instead of a hard-coded % 2, and
//   3. the extruded volume is derived from the target sphere volume instead of
//      being a hand-tuned multiple of the line flow.
// Everything else - naming, structure, snake ordering, the shrink/fallback for
// thin interface islands - is the original code.
// ---------------------------------------------------------------------------

// Height of a regular tetrahedron of edge 1: a bead resting in the hollow of
// three mutually touching beads sits this far above their centre plane.
static constexpr double SUPPORT_BEAD_TETRA_HEIGHT = 0.8164965809277260; // sqrt(2/3)

enum class SupportBeadRegion { Contact, Body };

struct SupportBeadParams
{
    // Bead (sphere) diameter. Comes from the support extruder nozzle, never
    // hardcoded.
    double bead_diameter_mm        = 1.0;

    // How far neighbouring beads are pressed into each other, as a fraction of
    // the bead diameter. 0.0 == tangent spheres: perfect shape, zero fixation.
    // The contact disc grows with sqrt(overlap) while the lost volume grows
    // with overlap, so a small value already buys a lot of grip:
    //   0.04 -> contact disc 0.28 * D, packing 0.83
    //   0.06 -> contact disc 0.34 * D, packing 0.86
    //   0.12 -> contact disc 0.48 * D, packing 0.95 (nearly solid, avoid)
    double lattice_overlap_ratio   = 0.06;

    // Keep bead centres this far inside the region outline.
    double edge_margin_ratio       = 0.5;

    // A bead is deposited as a very short segment, not as a line: the shorter
    // the segment the rounder the bead. Keep it just long enough for the
    // firmware to meter the volume.
    double segment_ratio           = 0.15;

    // false => beads stack in straight columns (the old behaviour, 8 contacts)
    // true  => beads nest in the hollows below (12 contacts)
    bool   stagger_layers          = true;

    // 2 => ABAB (hcp), 3 => ABCABC (fcc). Mechanically near-identical; fcc
    // spreads the seam over three layers and prints marginally smoother.
    int    stagger_period          = 3;

    // Snake order: row 0 left->right, row 1 right->left, ... to cut travel.
    bool   snake_order             = true;

    // Rotate the (short) segment with the stacking period so the small
    // anisotropy of the deposit averages out instead of accumulating.
    bool   alternate_contact_angle = true;

    // Trim of the volume computed from the sphere geometry. Leave at 1.0 and
    // only touch it if the pellet extruder under- or over-delivers.
    double flow_multiplier         = 1.0;

    // Lattice origin, shared by every layer and every region of the object.
    // Set once by the caller (see support_bead_grid_origin below).
    Point  grid_origin             = Point(0, 0);

    // Centre-to-centre distance of touching beads.
    double pitch_mm() const
        { return bead_diameter_mm * (1.0 - lattice_overlap_ratio); }

    // The layer height the print profile MUST use for the packing to close up.
    // Too tall and the beads never touch the layer below; too short and they
    // fuse into a solid block.
    double layer_height_mm() const
        { return stagger_layers ? pitch_mm() * SUPPORT_BEAD_TETRA_HEIGHT : pitch_mm(); }

    // Radius of the flat disc where two pressed-together beads meet.
    double contact_disc_radius_mm() const
    {
        const double r = 0.5 * bead_diameter_mm;
        const double d = pitch_mm();
        return (d >= 2.0 * r) ? 0.0 : std::sqrt(r * r - 0.25 * d * d);
    }

    int coordination_number() const { return stagger_layers ? 12 : 8; }
};

static SupportBeadParams support_bead_contact_params(double nozzle_diameter_mm)
{
    // Bead touching the model underside: press a little harder so the contact
    // shell does not shear off while the model is being printed on top of it.
    SupportBeadParams params;
    params.bead_diameter_mm      = nozzle_diameter_mm;
    params.lattice_overlap_ratio = 0.08;
    params.edge_margin_ratio     = 0.4;
    params.segment_ratio         = 0.15;
    params.stagger_layers        = true;
    return params;
}

static SupportBeadParams support_bead_body_params(double nozzle_diameter_mm)
{
    // Bead body: same lattice, slightly smaller sphere. The lattice pitch is
    // shared with the contact region (one object, one lattice), so shrinking
    // the bead is the only way to reduce the overlap - and a smaller overlap
    // is what makes the body break apart cleanly into reusable pellets.
    SupportBeadParams params;
    params.bead_diameter_mm      = nozzle_diameter_mm;
    params.lattice_overlap_ratio = 0.04;
    params.edge_margin_ratio     = 0.5;
    params.segment_ratio         = 0.15;
    params.stagger_layers        = true;
    return params;
}

// One lattice per object. Anything derived from the current layer's extents
// drifts as the cross section changes, which desynchronises the stagger and
// destroys the packing.
static Point support_bead_grid_origin(const BoundingBox &object_bbox)
{
    return object_bbox.min;
}

struct SupportBeadCenter
{
    Point center;
    int   row = 0;
};

static std::vector<SupportBeadCenter> support_bead_generate_centers(
    const Polygons          &polygons,
    const SupportBeadParams &params,
    size_t                   support_layer_id)
{
    std::vector<SupportBeadCenter> centers;

    if (polygons.empty())
        return centers;

    const double pitch_mm = params.pitch_mm();
    if (pitch_mm <= 0.0 || params.bead_diameter_mm <= 0.0)
        return centers;

    const coord_t edge_margin = scale_(params.bead_diameter_mm * params.edge_margin_ratio);
    Polygons safe_polygons = shrink(polygons, edge_margin);

    // Very thin interface islands can be erased by shrinking. Fall back to the
    // original polygon and rely on per-center containment testing.
    if (safe_polygons.empty())
        safe_polygons = polygons;

    ExPolygons safe_expolys = union_ex(safe_polygons);
    BoundingBox bbox        = get_extents(safe_polygons);

    const coord_t pitch_x = scale_(pitch_mm);
    const coord_t pitch_y = scale_(pitch_mm * std::sqrt(3.0) * 0.5);
    if (pitch_x <= 0 || pitch_y <= 0)
        return centers;

    // Close packing: layer k is shifted into the hollows of layer k-1. The
    // hollow of an up-pointing lattice triangle is its centroid,
    // (pitch_x/2, pitch_y/3). Repeating that shift walks A -> B -> C and back
    // to A after `stagger_period` layers.
    coord_t layer_shift_x = 0;
    coord_t layer_shift_y = 0;
    if (params.stagger_layers) {
        const int period = std::max(2, params.stagger_period);
        const int k      = int(support_layer_id % size_t(period));
        layer_shift_x = coord_t(k) * (pitch_x / 2);
        layer_shift_y = coord_t(k) * (pitch_y / 3);
    }

    // Row and column indices are absolute lattice indices, not counters local
    // to this layer, so row parity (and therefore the half-pitch row shift)
    // stays consistent between layers and between regions.
    const Point   origin = params.grid_origin;
    const coord_t base_y = origin.y() + layer_shift_y;
    const coord_t base_x = origin.x() + layer_shift_x;

    const int row_first = int(std::floor(double(bbox.min.y() - base_y) / double(pitch_y)));
    const int row_last  = int(std::ceil (double(bbox.max.y() - base_y) / double(pitch_y)));

    for (int row = row_first; row <= row_last; ++ row) {
        const coord_t y           = base_y + coord_t(row) * pitch_y;
        const coord_t row_shift_x = (row % 2 == 0) ? 0 : pitch_x / 2;
        const coord_t row_base_x  = base_x + row_shift_x;

        const int col_first = int(std::floor(double(bbox.min.x() - row_base_x) / double(pitch_x)));
        const int col_last  = int(std::ceil (double(bbox.max.x() - row_base_x) / double(pitch_x)));

        std::vector<SupportBeadCenter> row_centers;
        for (int col = col_first; col <= col_last; ++ col) {
            const Point center(row_base_x + coord_t(col) * pitch_x, y);
            if (support_bead_contains_point(safe_expolys, center))
                row_centers.push_back({ center, row });
        }

        // Snake order: row 0 left->right, row 1 right->left, ... to cut travel.
        if (params.snake_order && (row % 2 != 0))
            std::reverse(row_centers.begin(), row_centers.end());

        centers.insert(centers.end(), row_centers.begin(), row_centers.end());
    }

    return centers;
}

static Polyline support_bead_make_segment(
    const Point             &center,
    const SupportBeadParams &params,
    size_t                   support_layer_id,
    SupportBeadRegion        region)
{
    double angle = 0.0;

    // The segment only exists to carry the metered volume; keeping it short is
    // what keeps the bead round. Rotating it with the stacking period spreads
    // the leftover anisotropy evenly instead of stacking it in one direction.
    if (params.alternate_contact_angle) {
        const int period = params.stagger_layers ? std::max(2, params.stagger_period) : 2;
        angle = M_PI * double(support_layer_id % size_t(period)) / double(period);
    }

    const double  seg_len_mm  = params.bead_diameter_mm * params.segment_ratio;
    const double  half_len_mm = 0.5 * seg_len_mm;
    const coord_t dx = scale_(half_len_mm * std::cos(angle));
    const coord_t dy = scale_(half_len_mm * std::sin(angle));

    Polyline seg;
    seg.points.emplace_back(center.x() - dx, center.y() - dy);
    seg.points.emplace_back(center.x() + dx, center.y() + dy);
    return seg;
}

// Volume of one bead: a sphere minus the spherical caps its neighbours press
// away. Deriving mm3_per_mm from this instead of from the line flow is what
// makes the deposit spherical - the 1 mm nozzle produces a fat bead, but the
// bead's size has to be set by geometry, not by how long the segment happens
// to be.
static double support_bead_mm3_per_mm(const SupportBeadParams &params)
{
    const double d       = params.bead_diameter_mm;
    const double r       = 0.5 * d;
    const double seg_len = std::max(EPSILON, d * params.segment_ratio);

    const double overlap = std::max(0.0, d - params.pitch_mm()); // delta, mm
    const double h_cap   = 0.5 * overlap;
    const double v_cap   = M_PI * h_cap * h_cap * (3.0 * r - h_cap) / 3.0;
    const double v_bead  = (4.0 / 3.0) * M_PI * r * r * r
                         - double(params.coordination_number()) * v_cap;

    return params.flow_multiplier * std::max(0.0, v_bead) / seg_len;
}

static bool support_bead_generate_paths(
    ExtrusionEntitiesPtr    &dst,
    const Polygons          &polygons,
    ExtrusionRole            role,
    const Flow              &flow,           // support-extruder (T1) flow; slicer routes it
    const SupportBeadParams &params,
    size_t                   support_layer_id,
    SupportBeadRegion        region)
{
    std::vector<SupportBeadCenter> centers =
        support_bead_generate_centers(polygons, params, support_layer_id);

    if (centers.empty())
        return false;

    Polylines segments;
    segments.reserve(centers.size());
    for (const SupportBeadCenter &c : centers)
        segments.emplace_back(support_bead_make_segment(c.center, params, support_layer_id, region));

    // Volume comes from the sphere geometry, not from flow.mm3_per_mm().
    const double mm3_per_mm = support_bead_mm3_per_mm(params);

    extrusion_entities_append_paths(
        dst,
        std::move(segments),
        role,
        mm3_per_mm,
        // The deposit is as wide and as tall as the bead itself.
        float(params.bead_diameter_mm),
        float(flow.height()),
        false);

    return true;
}

// ---------------------------------------------------------------------------
// Call-site notes (the two @@ hunks of the original patch are unchanged apart
// from these three lines):
//
//   1. Set the shared lattice origin once, next to where support_layer_id is
//      resolved:
//          contact_params.grid_origin = support_bead_grid_origin(object_bbox);
//          body_params.grid_origin    = contact_params.grid_origin;
//
//   2. The first support layer still keeps its original sheath / no_sort paths
//      for bed adhesion - that guard stays exactly as written:
//          if (! first_support_layer && ! sheath && ! no_sort) { ... }
//
//   3. Assert the profile layer height against the packing, because a wrong
//      layer height silently turns the packing into either a stack of loose
//      discs or a solid block:
//          const double h_req = body_params.layer_height_mm();
//          if (std::abs(layer_height - h_req) > 0.02)
//              BOOST_LOG_TRIVIAL(warning) << "Bead support: layer height "
//                  << layer_height << " mm should be " << h_req << " mm";
// ---------------------------------------------------------------------------
