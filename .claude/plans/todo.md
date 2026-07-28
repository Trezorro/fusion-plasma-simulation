# Todo dump for todos that need to be picked up later
I would have prompted you earlier with this but for any reason I could not.

- [done] The rollout simulation view legend is now grouped by "Ground truth" vs each
  individual sample, one legendgroup spanning all channel traces + the label row, so a
  single click hides/shows a whole sample or the real trace at once. Original history
  (left of the rollout start) is not part of any legend group, so it always shows.
  Renamed 'X real' to 'Ground truth' as a single group. Legend moved to the right,
  below the rollout dropdown (both x=1.02). Colors were already explicit per trace in
  code (line.color set on every trace); could not reproduce the "always orange"
  symptom from the code as it stood, flag again if it's still visible.
  - [done] The browser now overlays up to `rollout.plot_samples` (default 3) stochastic
    samples per starting point in one dropdown entry, via src.rollout.build_rollout_groups.
