#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is $ds->entropy([1]), 0, 'entropy';
is $ds->entropy([0.5, 0.5]), 1, 'entropy';
my $got = $ds->entropy([0.25, 0.75]);
is sprintf('%.4f', $got), '0.8113', 'entropy';

$got = $ds->class_probablities(['a','b']);
is_deeply $got, [0.5, 0.5], 'class_probablities';

is $ds->data_entropy(['a']), 0, 'data_entropy';
is $ds->data_entropy([0, 1]), 1, 'data_entropy';
is $ds->data_entropy([1,2,3,4]), 2, 'data_entropy';

my @inputs = (
    { level => 'Senior', lang => 'Java',   tweets => 0, phd => 0, did_well => 0 },
    { level => 'Senior', lang => 'Java',   tweets => 0, phd => 1, did_well => 0 },
    { level => 'Mid',    lang => 'Python', tweets => 0, phd => 0, did_well => 1 },
    { level => 'Junior', lang => 'Python', tweets => 0, phd => 0, did_well => 1 },
    { level => 'Junior', lang => 'R',      tweets => 1, phd => 0, did_well => 1 },
    { level => 'Junior', lang => 'R',      tweets => 1, phd => 1, did_well => 0 },
    { level => 'Mid',    lang => 'R',      tweets => 1, phd => 1, did_well => 1 },
    { level => 'Senior', lang => 'Python', tweets => 0, phd => 0, did_well => 0 },
    { level => 'Senior', lang => 'R',      tweets => 1, phd => 0, did_well => 1 },
    { level => 'Junior', lang => 'Python', tweets => 1, phd => 0, did_well => 1 },
    { level => 'Senior', lang => 'Python', tweets => 1, phd => 1, did_well => 1 },
    { level => 'Mid',    lang => 'Python', tweets => 0, phd => 1, did_well => 1 },
    { level => 'Mid',    lang => 'Java',   tweets => 1, phd => 0, did_well => 1 },
    { level => 'Junior', lang => 'Python', tweets => 0, phd => 1, did_well => 0 },
);

my $expect = {
  level    => 'Senior',
  lang     => 'Java',
  tweets   => 0,
  phd      => 0,
  did_well => 0,
};
my %h = $ds->partition_by(\@inputs, 'level');
is_deeply $h{Senior}[0], $expect, 'partition_by';

$got = $ds->partition_entropy_by(\@inputs, 'level', 'did_well');
is sprintf('%.4f', $got), '0.6935', 'partition_entropy_by';
$got = $ds->partition_entropy_by(\@inputs, 'lang', 'did_well');
is sprintf('%.4f', $got), '0.8601', 'partition_entropy_by';
$got = $ds->partition_entropy_by(\@inputs, 'tweets', 'did_well');
is sprintf('%.4f', $got), '0.7885', 'partition_entropy_by';
$got = $ds->partition_entropy_by(\@inputs, 'phd', 'did_well');
is sprintf('%.4f', $got), '0.8922', 'partition_entropy_by';

my @senior_inputs = map { $_ } grep { $_->{level} eq 'Senior' } @inputs;

$got = $ds->partition_entropy_by(\@senior_inputs, 'lang', 'did_well');
is sprintf('%.4f', $got), '0.4000', 'partition_entropy_by';
$got = $ds->partition_entropy_by(\@senior_inputs, 'tweets', 'did_well');
is sprintf('%.4f', $got), '0.0000', 'partition_entropy_by';
$got = $ds->partition_entropy_by(\@senior_inputs, 'phd', 'did_well');
is sprintf('%.4f', $got), '0.9510', 'partition_entropy_by';

$expect = {
    subtrees => {
        Senior => {
            attribute => 'tweets',
            subtrees => { 0 => { value => 0 }, 1 => { value => 1 } },
            default_value => 0,
        },
        Mid => { value => 1 },
        Junior => {
            attribute => 'phd',
            subtrees => { 0 => { value => 1 }, 1 => { value => 0 } },
            default_value => 1,
        }
    },
    attribute => 'level',
    default_value => 1,
};
my $tree = $ds->build_tree_id3(\@inputs, ['level', 'lang', 'tweets', 'phd'], 'did_well');
is_deeply $tree, $expect, 'build_tree_id3';

ok $ds->classify($tree, {level => 'Junior', lang => 'Java', tweets => 1, phd => 0}), 'classify';
ok ! $ds->classify($tree, {level => 'Junior', lang => 'Java', tweets => 1, phd => 1}), 'classify';
ok $ds->classify($tree, {level => 'Intern', lang => 'Java', tweets => 1, phd => 1}), 'classify';

done_testing();
