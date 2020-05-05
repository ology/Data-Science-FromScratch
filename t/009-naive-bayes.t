#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use List::Util qw(sum0);

use_ok 'Data::MachineLearning::Elements';

my $ds = new_ok 'Data::MachineLearning::Elements';

my $got = $ds->tokenize('Little Mary had a little lamb.');
is_deeply $got, [qw(a had lamb little mary)], 'tokenize';

my @messages = (
    { text => 'spam rules', is_spam => 1 },
    { text => 'ham rules', is_spam => 0 },
    { text => 'hello ham', is_spam => 0 },
);
$ds->train(@messages);

is_deeply [sort keys %{ $ds->tokens }], [qw(ham hello rules spam)], 'tokens';
is $ds->spam_messages, 1, 'spam_messages';
is $ds->ham_messages, 2, 'ham_messages';
is_deeply $ds->token_spam_counts, { spam => 1, rules => 1 }, 'token_spam_counts';
is_deeply $ds->token_ham_counts, { ham => 2, rules => 1, hello => 1 }, 'token_ham_counts';

my @probs_if_spam = (
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
);
my @probs_if_ham = (
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
);

my $p_if_spam = exp sum0(map { log $_ } @probs_if_spam);
my $p_if_ham = exp sum0(map { log $_ } @probs_if_ham);

$got = $ds->nb_predict('hello spam');
is $got, $p_if_spam / ($p_if_spam + $p_if_ham), 'nb_predict';

done_testing();
