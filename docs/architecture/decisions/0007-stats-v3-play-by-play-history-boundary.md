# ADR-0007: Stats V3 Play-By-Play History Boundary

<p class="adr-status"><strong>Decision status</strong><span>Accepted</span></p>

## Context

[ADR-0006](0006-historical-stats-v3-first.md) selected NBA Stats V3 as the
historical source. The usable history is bounded by `playbyplayv3`, rather than
by a schedule row or box score alone: possession segmentation and lineup
reconstruction require a non-empty event stream.

Direct source probes found populated V3 play-by-play responses from 1996-97
onward. Equivalent 1995-96 and earlier probes returned valid V3 envelopes with
empty `game.actions` arrays. Some older `scheduleleaguev2` responses also
contain unplayed exhibition or if-necessary placeholders with status other than
final and both team identities set to zero.

## Decision

- Treat 1996-97 as the practical lower boundary for V3-based play-by-play acquisition.
- Do not treat earlier V3 box-score availability as possession-model coverage.
- Exclude a schedule row only when it is non-final and has no identifiable home
  and away teams.
- Retain final games and every game with valid team identities under the normal
  strict catalog contract.
- Archive regular-season play-by-play before optional endpoint or postseason
  expansion, with raw files as the resume boundary.

## Consequences

The acquisition target is approximately 30 seasons of regular-season event
data, 1996-97 through the current completed season. The boundary is source- and
endpoint-specific: a different NBA-owned feed could extend earlier history in a
future decision, but it must be evaluated independently and must not be merged
silently with the V3 archive.

The catalog can contain identified preseason and All-Star games for provenance.
They are not selected by the historical regular-season archive flow unless an
explicit season type is requested.
