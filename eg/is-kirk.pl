#!/usr/bin/env perl
use strict;
use warnings;

# Did Kirk say it? - Naive Bayes analysis of Kirk, Spock and McCoy lines.
# spam = is kirk. ham = is not kirk
# Star Trek scripts used:
# https://github.com/ology/Machine-Learning/blob/master/Kirk-Spock-McCoy.zip

use Data::Science::FromScratch;
use File::Slurper qw(read_text);
use Lingua::EN::Sentence qw(get_sentences);

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

print "Training on messages...\n";
$ds->train(@messages);

print 'Kirk said ', $ds->spam_messages, " words.\n";
print 'Spock or McCoy said ', $ds->ham_messages, " words.\n";

print "Probability that Kirk said,\n";
for my $text (
    'Shall we pick some flowers, Doctor?',                      # Kirk
    'Vulcan has no moon, Miss Uhura.',                          # Spock
    'Is that how you get girls to like you, by bribing them?',  # McCoy
    'Give me warp nine, Scotty.',                               # Fake Kirk
    'That is highly illogical.',                                # Fake Spock
    "He's dead, Jim.",                                          # Fake McCoy
) {
    my $prediction = $ds->nb_predict($text);
    printf qq/\t%.4f <= "%s"\n/, $prediction, $text;
}
