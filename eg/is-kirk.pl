#!/usr/bin/env perl
use strict;
use warnings;

# Did Kirk say it? - Naive Bayes analysis of Kirk, Spock and McCoy lines.
# Where spam is Kirk and ham is not Kirk.
# Star Trek scripts used:
# https://github.com/ology/Machine-Learning/blob/master/Kirk-Spock-McCoy.zip

use Data::Science::FromScratch;
use File::Slurper qw(read_text);
use Lingua::EN::Sentence qw(get_sentences);

# How big should the traing set be?
my $split = shift || 0.4;
# Probability threshold (confidence) that Kirk said it
my $threshold = shift || .7;
# Provide a custom statement to process
my $statement = shift || 'To be or not to be. That is the question.';

# Unziped Star Trek scripts in this location
my $path = $ENV{HOME} . '/Documents/lit/Kirk-Spock-McCoy/';

print "Gathering messages...\n";
my @messages;

# Process the lines of each file
for my $name (qw(kirk spock mccoy)) {
    my $file = $path . $name . '.txt';

    my $content   = read_text($file);
    my $sentences = get_sentences($content);

    for my $sentence (@$sentences) {
        $sentence =~ s/\(.+?\)//; # Remove action instructions

        next if $sentence =~ /\(/; # Skip broken actions

        $sentence =~ s/\n+/ /g; # Make the sentence a single line
        $sentence =~ s/^\s*//; # Trim whitespace
        $sentence =~ s/\s*$//; # Trim whitespace

        next unless $sentence =~ /\w/;
    #    print $sentence, "\n\n";

        # The processed messages are a list of 2-key hashrefs
        push @messages, { text => $sentence, is_spam => $name eq 'kirk' ? 1 : 0 };
    }
}

# Invoke the data science library
my $ds = Data::Science::FromScratch->new;

my ($train, $test) = $ds->split_data($split, @messages);

print "Training on messages...\n";
$ds->train(@$train);

print 'Kirk said ', $ds->spam_messages, " sentences.\n";
print 'Spock and McCoy said ', $ds->ham_messages, " sentences.\n";

print "Probability that Kirk said,\n";
for my $text (
    'Shall we pick some flowers, Doctor?',                      # Kirk
    'Vulcan has no moon, Miss Uhura.',                          # Spock
    'Is that how you get girls to like you, by bribing them?',  # McCoy
    'Give me warp nine, Scotty.',                               # Fake Kirk
    'That is highly illogical.',                                # Fake Spock
    "He's dead, Jim.",                                          # Fake McCoy
    $statement
) {
    next unless $text;
    my $prediction = $ds->nb_predict($text);
    printf qq/\t%.4f = "%s"\n/, $prediction, $text;
}

print "Computing accuracy, etc...\n";
my ($tp, $fp, $fn, $tn) = (0,0,0,0);
my %confusion_matrix;
for my $i (@$test) {
    my $predicted = $ds->nb_predict($i->{text});
    my $true_pos  =  $i->{is_spam} && $predicted >= $threshold ? 1 : 0;
    my $false_pos = !$i->{is_spam} && $predicted >= $threshold ? 1 : 0;
    my $false_neg =  $i->{is_spam} && $predicted <  $threshold ? 1 : 0;
    my $true_neg  = !$i->{is_spam} && $predicted <  $threshold ? 1 : 0;
    $confusion_matrix{"$true_pos,$false_pos,$false_neg,$true_neg"}++;
    $tp += $true_pos;
    $fp += $false_pos;
    $fn += $false_neg;
    $tn += $true_neg;
}
print "Confusion matrix:\n";
print join("\n", map { "\t$_ => $confusion_matrix{$_}" } sort keys %confusion_matrix), "\n";
printf "Accuracy = %.4f\nPrecision = %.4f\nRecall = %.4f\nf1_score = %.4f\n",
    $ds->accuracy($tp, $fp, $fn, $tn),
    $ds->precision($tp, $fp, $fn, $tn),
    $ds->recall($tp, $fp, $fn, $tn),
    $ds->f1_score($tp, $fp, $fn, $tn);
