# V3 user split design

## Session distribution

| Session | Images | Laptop | Keyboard | Cup | Negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| S01 | 28 | 13 | 0 | 18 | 3 |
| S02 | 60 | 19 | 19 | 29 | 21 |
| S03 | 8 | 7 | 0 | 0 | 1 |
| S04 | 18 | 0 | 2 | 15 | 6 |
| S05 | 5 | 5 | 0 | 4 | 0 |

## Old split

Train: S01, S02 — 88 images, 32/19/47 laptop/keyboard/cup.
Val: S05 — 5 images, 5/0/4.
Test: S03, S04 — 26 images, 7/2/15 after the annotation correction (previously 7/2/14).

## Enumeration result

All 150 non-empty assignments were enumerated; 24 satisfy mandatory three-class Train and Test coverage.
No assignment can cover all three classes in Train, Val, and Test: keyboard occurs only in S02 and S04, but three disjoint splits require three keyboard-bearing sessions.

Ranking follows the requested priority lexicographically: Val coverage, Test minimum >=5, Val minimum >=3, Test balance, Val balance, Train >=60%, then 70/15/15 proximity.

## Top five candidates

| Rank | Train | Val | Test | Train L/K/C | Val L/K/C | Test L/K/C | Test range | Val range |
| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 1 | S01,S03,S04 | S05 | S02 | 20/2/33 | 5/0/4 | 19/19/29 | 10 | 5 |
| 2 | S01,S04 | S05 | S02,S03 | 13/2/33 | 5/0/4 | 26/19/29 | 10 | 5 |
| 3 | S01,S04 | S03,S05 | S02 | 13/2/33 | 12/0/4 | 19/19/29 | 10 | 12 |
| 4 | S03,S04,S05 | S01 | S02 | 12/2/19 | 13/0/18 | 19/19/29 | 10 | 18 |
| 5 | S04,S05 | S01 | S02,S03 | 5/2/19 | 13/0/18 | 26/19/29 | 10 | 18 |

## Final split

Train: S01, S03, S04 (54 images).
Val: S05 (5 images).
Test: S02 (60 images).

This makes Test 19/19/29 and resolves the previous two-keyboard diagnostic weakness without splitting a session. Val remains 5/0/4 because the two keyboard sessions are required by Train and Test. The choice sacrifices the 60% Train image preference because coverage and Test/Val balance have higher declared priority.
