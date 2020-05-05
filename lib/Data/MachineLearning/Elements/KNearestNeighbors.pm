package Data::MachineLearning::Elements::KNearestNeighbors;

use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ds = Data::MachineLearning::Elements->new;

  my $y = $ds->majority_vote(qw(a b c b a)); # b

  $y = $ds->knn_classify(); #

=head1 METHODS

=head2 majority_vote

  $y = $ds->majority_vote(@labels);

=cut

sub majority_vote {
    my ($self, @labels) = @_;
    my %vote_counts;
    $vote_counts{$_}++
        for @labels;
    my $winner = (sort { $vote_counts{$b} <=> $vote_counts{$a} } keys %vote_counts)[0];
    my $winner_count = $vote_counts{$winner};
    my $num_winners = grep { $_ == $winner_count } values %vote_counts;
    if ($num_winners == 1) {
        return $winner;
    }
    else {
        return $self->majority_vote(@labels[0 .. @labels - 2]);
    }
}

=head2 knn_classify

  $y = $ds->knn_classify($k, $labeled_points, $new_point);

Where B<k> is an integer, B<labeled_points> is a list of hash
references,

  [ { point => $vector, label => $string }, { ... } ]

and the B<new_point> is a vector (array reference of numbers).

=cut

sub knn_classify {
    my ($self, $k, $labeled_points, $new_point) = @_;
    my @by_distance = sort {
        $self->distance($a->{point}, $new_point) <=> $self->distance($b->{point}, $new_point)
    } @$labeled_points;
    my @k_nearest_labels = map { $_->{label} } @by_distance[0 .. $k];
    return $self->majority_vote(@k_nearest_labels);
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/008-k-nearest-neighbors.t>

L<Moo::Role>

=cut
