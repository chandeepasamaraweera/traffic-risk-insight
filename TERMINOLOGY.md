# Terminology and Claim Boundaries

## Secondary crash

A secondary crash is a crash that occurs as a result of an earlier incident, either within the original incident scene or within the queue or backup produced by that incident. The phrase implies an operational relationship, not merely closeness in time and space.

## Primary incident

The earlier incident associated with the disturbance, incident scene, or queue in which a secondary crash occurs. The US Accidents event stream used in this study does not designate verified primary incidents for later crashes.

## Spatio-Temporally Proximate Collision (STPC)

A proxy label created by this research. An event is STPC-positive when at least one strictly earlier accident occurred within the selected spatial radius and within the preceding two hours. Candidate radii are experimental parameters.

STPC means **proximity to a prior accident**, not confirmation that the prior accident caused the later event.

## Proximate secondary traffic risk

The study's prediction target at the interpretive level: the relative next-hour risk that a region will contain an STPC-positive event. The phrase preserves the secondary-crash motivation while signalling that the available target is a proximity proxy.

## Region-hour

The unit of analysis. One observation represents one macro-region during one hour. A four-hour feature history is used to predict the following hour.

## Event-derived graph

A graph constructed from accident locations rather than a verified road network. KMeans macro-regions are nodes; distance-based k-nearest-neighbour relationships between region centroids provide edges.

## Native road topology

Observed transportation-network structure such as road segments, legal movements, ramps, intersections, carriageways, and direction. It is unavailable in the source event stream used here.

## AUC-PR and Average Precision

The implementation computes scikit-learn `average_precision_score`. The thesis and application report this result as AUC-PR. Documentation should use the precise formulation **Average Precision (reported as AUC-PR)** when implementation detail matters.

## Permitted claims

- The model ranks the STPC proxy on held-out region-hours.
- Accident streams contain limited temporal and contextual predictive information.
- The tested event-derived graph did not provide a consistent advantage across the core urban benchmark.
- Results are sensitive to spatial representation.

## Prohibited claims

- STPC-positive events are confirmed secondary crashes.
- A graph edge represents a verified traffic connection.
- Model output establishes causation.
- The system predicts individual crashes or identifies responsible drivers.
- The packaged application is a validated live traffic-management system.
