package Data::Science::FromScratch::DecisionTrees;

use List::Util qw(sum0);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $x = $ds->entropy([0.25, 0.75]); # 0.8113

  my $v = $ds->class_probablities(['a','b']); # [0.5, 0.5]

  $x = $ds->data_entropy([0,1]); # 1

  $x = $ds->partition_entropy(); # 

  my %h = $ds->partition_by(); # 

  $x = $ds->partition_entropy_by(); # 

  $v = $ds->classify($tree, $input); # TODO

  $x = $ds->build_tree_id3(); # TODO

=head1 METHODS

=head2 entropy

  $x = $ds->entropy($probablities);

=cut

sub entropy {
    my ($self, $probablities) = @_;
    return sum0(map { - $_ * _log2($_) } grep { $_ > 0 } @$probablities);
}

sub _log2 {
    my $n = shift;
    return log($n)/log(2);
}

=head2 class_probablities

  $x = $ds->class_probablities($labels);

=cut

sub class_probablities {
    my ($self, $labels) = @_;
    my %labels;
    for my $label (@$labels) {
        $labels{$label}++;
    }
    return [map { $_ / @$labels } values %labels];
}

=head2 data_entropy

  $x = $ds->data_entropy($labels);

=cut

sub data_entropy {
    my ($self, $labels) = @_;
    return $self->entropy($self->class_probablities($labels));
}

=head2 partition_entropy

  $x = $ds->partition_entropy($subsets);

=cut

sub partition_entropy {
    my ($self, $subsets) = @_;
    my $total_count = sum0(map { scalar @$_ } @$subsets);
    return sum0(map { $self->data_entropy($_) * @$_ / $total_count } @$subsets);
}

=head2 partition_by

  %h = $ds->partition_by($inputs, $attribute);

=cut

sub partition_by {
    my ($self, $inputs, $attribute) = @_;
    my %partitions;
    for my $input (@$inputs) {
        push @{ $partitions{ $input->{$attribute} } }, $input;
    }
    return %partitions;
}

=head2 partition_entropy_by

  $x = $ds->partition_entropy_by($inputs, $attribute, $label_attr);

=cut

sub partition_entropy_by {
    my ($self, $inputs, $attribute, $label_attr) = @_;
    my %partitions = $self->partition_by($inputs, $attribute);
    my @labels;
    for my $partition (values %partitions) {
        push @labels, [ map { $_->{$label_attr} } @$partition ];
    }
    return $self->partition_entropy(\@labels);
}

=head2 classify

  $x = $ds->classify($tree, $input);

TODO

=cut

sub classify {
    my ($self, $tree, $input) = @_;
}

=head2 build_tree_id3

  $x = $ds->build_tree_id3($inputs, $split_attributes, $target_attribute);

TODO

=cut

sub build_tree_id3 {
    my ($self, $inputs, $split_attributes, $target_attribute) = @_;
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/013-decision-trees.t>

L<List::Util>

L<Moo::Role>

=cut
