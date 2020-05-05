package Data::MachineLearning::Elements::LinearAlgebra;

use List::Util qw(all);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my $v = $ml->vector_sum([1,2,3], [4,5,6]); # [5,7,9]

  $v = $ml->vector_subtract([5,7,9], [4,5,6]); # [1,2,3]

  $v = $ml->scalar_multiply(2, [1,2,3]); # [2,4,6]

  $v = $ml->vector_mean([1,2], [3,4], [5,6]); # [3,4]

  my $y = $ml->vector_dot([1,2,3], [4,5,6]); # 32

  $y = $ml->sum_of_squares([1,2,3]); # 14

  $y = $ml->magnitude([3,4]); # 5

  $y = $ml->distance([0,1], [1,1]); # 1

  $y = $ml->squared_distance([0,0], [2,2]); # 8

  $v = $ml->shape([[1,2,3], [4,5,6]]); # [2,3]

  $v = $ml->get_row([[1,2,3], [4,5,6]], 0); # [1,2,3]
  $v = $ml->get_col([[1,2,3], [4,5,6]], 0); # [1,4]

  my $m = $ml->make_matrix(2, 3, sub { 0 }); # [[0,0,0], [0,0,0]]
  $m = $ml->make_matrix(3, 3, sub { my ($i, $j) = @_; return $i == $j ? 1 : 0 });
     # [[1,0,0], [0,1,0], [0,0,1]]


=head1 VECTOR METHODS

=head2 vector_sum

  $v = $ml->vector_sum(@vectors);

=cut

sub vector_sum {
    my ($self, @vectors) = @_;
    die 'Vectors must be of equal length'
        unless all { @{ $vectors[0] } == @$_ } @vectors;
    my @v;
    for my $i (0 .. @{ $vectors[0] } - 1) {
        my $x;
        for my $v (@vectors) {
            if ($x) {
                $x += $v->[$i];
            }
            else {
                $x = $v->[$i];
            }
        }
        push @v, $x;
    }
    return \@v;
}

=head2 vector_subtract

  $v = $ml->vector_subtract(@vectors);

=cut

sub vector_subtract {
    my ($self, @vectors) = @_;
    die 'Vectors must be of equal length'
        unless all { @{ $vectors[0] } == @$_ } @vectors;
    my @v;
    for my $i (0 .. @{ $vectors[0] } - 1) {
        my $x;
        for my $v (@vectors) {
            if ($x) {
                $x -= $v->[$i];
            }
            else {
                $x = $v->[$i];
            }
        }
        push @v, $x;
    }
    return \@v;
}

=head2 scalar_multiply

  $v = $ml->scalar_multiply($n, $vector);

=cut

sub scalar_multiply {
    my ($self, $n, $vector) = @_;
    my @v;
    for my $i (@$vector) {
        push @v, $n * $i;
    }
    return \@v;
}

=head2 vector_mean

  $v = $ml->vector_mean(@vectors);

Compute the componentwise means of a list of vectors.

=cut

sub vector_mean {
    my ($self, @vectors) = @_;
    my $n = 1 / @vectors;
    my $sum = $self->vector_sum(@vectors);
    return $self->scalar_multiply($n, $sum);
}

=head2 vector_dot

  $y = $ml->vector_dot(@vectors);

=cut

sub vector_dot {
    my ($self, @vectors) = @_;
    die 'Vectors must be of equal length'
        unless all { @{ $vectors[0] } == @$_ } @vectors;
    my $dot;
    for my $i (0 .. @{ $vectors[0] } - 1) {
        my $x;
        for my $v (@vectors) {
            if ($x) {
                $x *= $v->[$i];
            }
            else {
                $x = $v->[$i];
            }
        }
        $dot += $x;
    }
    return $dot;
}

=head2 sum_of_squares

  $y = $ml->sum_of_squares($vector);

=cut

sub sum_of_squares {
    my ($self, $vector) = @_;
    return $self->vector_dot($vector, $vector);
}

=head2 magnitude

  $y = $ml->magnitude($vector);

=cut

sub magnitude {
    my ($self, $vector) = @_;
    return sqrt $self->sum_of_squares($vector);
}

=head2 distance

  $y = $ml->distance($vector1, $vector2);

=cut

sub distance {
    my ($self, @vectors) = @_;
    return $self->magnitude($self->vector_subtract(@vectors));
}

=head2 squared_distance

  $y = $ml->squared_distance($vector1, $vector2);

=cut

sub squared_distance {
    my ($self, @vectors) = @_;
    return $self->sum_of_squares($self->vector_subtract(@vectors));
}

=head1 MATRIX METHODS

=head2 shape

  $v = $ml->shape($matrix);

=cut

sub shape {
    my ($self, $matrix) = @_;
    my $rows = @$matrix;
    my $cols = $matrix->[0] ? @{ $matrix->[0] } : 0;
    return [ $rows, $cols ];
}

=head2 get_row

  $v = $ml->get_row($matrix, $i);

=cut

sub get_row {
    my ($self, $matrix, $n) = @_;
    return $matrix->[$n];
}

=head2 get_col

  $v = $ml->get_col($matrix, $i);

=cut

sub get_col {
    my ($self, $matrix, $n) = @_;
    my @v;
    for my $m (@$matrix) {
        push @v, $m->[$n];
    }
    return \@v;
}

=head2 make_matrix

  $m = $ml->make_matrix($rows, $cols, $entry_fn);

=cut

sub make_matrix {
    my ($self, $rows, $cols, $entry_fn) = @_;
    my @matrix;
    for my $i (1 .. $rows) {
        my @row;
        for my $j (1 .. $cols) {
            push @row, $entry_fn->($i, $j);
        }
        push @matrix, \@row;
    }
    return \@matrix;
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/001-linear-algebra.t>

L<List::Util>

L<Moo::Role>

=cut
