# Test Words

This file records the word samples used by the tests and local simulation.

The tests do not load this file. The word lists are still defined inside the
test functions so each test can be read on its own. This file exists as a quick
reference for the data behind those tests.

## Bigram Extraction Cases

These strings are used to check how `bigrams` handles normal words, uppercase
letters, punctuation, short inputs, and non-ASCII characters.

```text
Thing
A!b?cd
café
<empty string>
a
```

## Selector Unit Cases

These small word lists are used to test selector behavior without running the
full simulation.

```text
alpha
bravo
charlie
plain
thing
think
```

## Evaluation Metric Cases

These words are used to check repeat and uniqueness metrics.

```text
one
two
```

## Small Parameter Search Sample

This compact sample is used by `TestParameterSearchReturnsBestCandidatesFirst`.
It includes words containing the synthetic weak bigrams `th`, `ng`, `mb`, and
`qu`, plus a few neutral words.

```text
thing
other
bring
strong
bemba
mumba
quick
quiet
plain
table
later
sound
```

## Default Simulation Sample

`TestIndexedSelectionNeedsExploration` uses the same `defaultWords()` sample as
the CLI experiment. The current list is:

```text
about
above
after
again
along
among
angle
answer
another
around
because
before
begin
being
below
between
better
beyond
branch
bring
called
cannot
change
charge
check
choose
church
clear
close
could
danger
degree
different
during
early
earth
either
enough
every
example
family
father
figure
final
follow
friend
front
general
given
going
great
group
hand
happen
having
heard
heart
heavy
house
human
important
include
inside
instead
language
large
later
learn
letter
little
machine
matter
maybe
member
method
middle
might
minute
money
mother
number
often
other
paper
people
person
place
point
possible
present
question
quick
quiet
quite
rather
reason
record
right
round
school
second
should
simple
small
something
sound
state
still
strong
system
table
taken
their
there
thing
think
those
though
three
through
together
toward
under
until
using
water
where
which
while
without
world
would
write
young
zambia
bemba
ubuntu
mumba
chomba
kalumba
ngoma
ngulu
thing
bring
string
strong
queen
quote
query
quite
```
