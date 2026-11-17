[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/HeyRdzy/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/HeyRdzy/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01

Forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), for
[make.fun's gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

`main.py` (Python 3.13, standard library only) reads one case file
(`/input/<id>.in`) and writes one forecast (`/output/<id>.out`): for each
of hours 2–24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent viewers. No network, no model calls, deterministic (two runs
write byte-identical output).

## Method

The model is fitted at run time from the case's own comparables — the past
big launches the case carries — never from a hardcoded list of names or
numbers. For each hour h = 2..24, ordinary least squares over those
launches:

```
log(viewers_h) = a_h + b_h * log(hour-1 print)
```

and the five forecast quantiles come from the fitted value plus the
empirical quantiles of the fit's residuals. The slope b_h is ~1 at hour 2
(the next hour looks like the last) and falls toward 0 by the evening (the
peak is set by the launch's scale, not by hour 1), so the model anchors on
hour 1 where hour 1 is informative and regresses to the launch-day shape
where it is not.

Only launches whose hour-1 print is at least 5,000 viewers join the fit.
Launches with a tiny hour 1 (a few hundred viewers at the New Zealand
unlock) still peaked at hundreds of thousands later in the day, so they
break the log-linear relation and flatten every slope; leaving them out is
what makes the evening hours scale with the launch instead of collapsing
to the field median.

## Validation

Scored with the competition's own pinball loss on the log scale, against
the practice cases' answers (SullyGnome hourly averages):

| model | mean practice loss (4 cases) |
|---|---|
| house baseline (persistence) | 0.387 |
| this entry | **0.226** |

Per case: Wolverine 0.336, Red Dead Redemption 2 0.228, God of War
Ragnarök 0.150, Marvel's Spider-Man 2 0.190. Leave-one-out over the
comparables: 0.21 mean loss on big launches (hour-1 print above 150k),
0.27 on medium ones — both well below the baseline in the same test.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
