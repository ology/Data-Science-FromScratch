package Data::Science::FromScratch::DecisionTrees;

use List::Util qw(min sum0);
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

  $v = $ds->classify($tree, $input); #

  my $h = $ds->build_tree_id3(); #

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

=cut

sub classify {
    my ($self, $tree, $input) = @_;
    if (exists $tree->{value}) {
        return $tree->{value};
    }
    my $subtree_key = $input->{ $tree->{attribute} };
    if (!grep { $_ eq $subtree_key } keys %{ $tree->{subtrees} }) {
        return $tree->{default_value};
    }
    my $subtree = $tree->{subtrees}{$subtree_key};
    return $self->classify($subtree, $input);
}

=head2 build_tree_id3

  $h = $ds->build_tree_id3($inputs, $split_attributes, $target_attribute);

=cut

sub build_tree_id3 {
    my ($self, $inputs, $split_attributes, $target_attribute) = @_;
    my %label_counts;
    for my $input (@$inputs) {
        $label_counts{ $input->{$target_attribute} }++;
    }
    my $most_common_label = (sort { $label_counts{$b} <=> $label_counts{$a} } keys %label_counts)[0];
    if (keys(%label_counts) == 1 || !@$split_attributes) {
        return { value => $most_common_label };
    }
    my $split_entropy = sub {
        my ($attr) = @_;
        return $self->partition_entropy_by($inputs, $attr, $target_attribute);
    };
    my %attributes;
    for my $x (@$split_attributes) {
        $attributes{ $split_entropy->($x) } = $x;
    }
    my $min__attribute = min(keys %attributes);
    my $best_attribute = $attributes{$min__attribute};
    my %partitions = $self->partition_by($inputs, $best_attribute);
    my @new_attributes = map { $_ } grep { $_ ne $best_attribute } @$split_attributes;
    my $subtrees = {
        map { $_ => $self->build_tree_id3($partitions{$_}, \@new_attributes, $target_attribute) } keys %partitions
    };
    return {
        attribute     => $best_attribute,
        subtrees      => $subtrees,
        default_value => $most_common_label,
    };
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/013-decision-trees.t>

L<List::Util>

L<Moo::Role>

=cut
