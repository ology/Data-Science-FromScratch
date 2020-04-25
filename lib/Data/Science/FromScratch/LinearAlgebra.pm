package Data::Science::FromScratch::LinearAlgebra;

use List::Util qw(all);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $v = $ds->vector_sum([1,2,3], [4,5,6]); # [5,7,9]

  $v = $ds->vector_subtract([5,7,9], [4,5,6]); # [1,2,3]

  $v = $ds->scalar_multiply(2, [1,2,3]); # [2,4,6]

  $v = $ds->vector_mean([1,2], [3,4], [5,6]); # [3,4]

  my $x = $ds->vector_dot([1,2,3], [4,5,6]); # 32

  $x = $ds->sum_of_squares([1,2,3]); # 14

  $x = $ds->magnitude([3,4]); # 5

  $x = $ds->distance([0,1], [1,1]); # 1

  $v = $ds->shape([[1,2,3], [4,5,6]]); # [2,3]

  $v = $ds->get_row([[1,2,3], [4,5,6]], 0); # [1,2,3]
  $v = $ds->get_col([[1,2,3], [4,5,6]], 0); # [1,4]

  my $m = $ds->make_matrix(2, 3, sub { 0 }); # [[0,0,0], [0,0,0]]
  $m = $ds->make_matrix(3, 3, sub { my ($i, $j) = @_; return $i == $j ? 1 : 0 });
     # [[1,0,0], [0,1,0], [0,0,1]]


=head1 VECTOR METHODS

=head2 vector_sum

  $v = $ds->vector_sum(@vectors);

=cut

sub vector_sum {
    my ($self, @vectors) = @_;
    die 'Vectors must be of equal length'
        unless all { @{ $vectors[0] } == @$_ } @vectors;
    my @v;
    for my $i (0 .. @{ $vectors[0] } - 1) {
        my $res;
        for my $v (@vectors) {
            if ($res) {
                $res += $v->[$i];
            }
            else {
                $res = $v->[$i];
            }
        }
        push @v, $res;
    }
    return \@v;
}

=head2 vector_subtract

  $v = $ds->vector_subtract(@vectors);

=cut

sub vector_subtract {
    my ($self, @vectors) = @_;
    die 'Vectors must be of equal length'
        unless all { @{ $vectors[0] } == @$_ } @vectors;
    my @v;
    for my $i (0 .. @{ $vectors[0] } - 1) {
        my $res;
        for my $v (@vectors) {
            if ($res) {
                $res -= $v->[$i];
            }
            else {
                $res = $v->[$i];
            }
        }
        push @v, $res;
    }
    return \@v;
}

=head2 scalar_multiply

  $v = $ds->scalar_multiply($n, $vector);

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

  $v = $ds->vector_mean($n, $vector);

=cut

sub vector_mean {
    my ($self, @vectors) = @_;
    my $n = 1 / @vectors;
    my $sum = $self->vector_sum(@vectors);
    return $self->scalar_multiply($n, $sum);
}

=head2 vector_mean

  $x = $ds->vector_mean($n, $vector);

=cut

sub vector_dot {
    my ($self, @vectors) = @_;
    die 'Vectors must be of equal length'
        unless all { @{ $vectors[0] } == @$_ } @vectors;
    my $dot;
    for my $i (0 .. @{ $vectors[0] } - 1) {
        my $res;
        for my $v (@vectors) {
            if ($res) {
                $res *= $v->[$i];
            }
            else {
                $res = $v->[$i];
            }
        }
        $dot += $res;
    }
    return $dot;
}

=head2 sum_of_squares

  $x = $ds->sum_of_squares($vector);

=cut

sub sum_of_squares {
    my ($self, $vector) = @_;
    return $self->vector_dot($vector, $vector);
}

=head2 magnitude

  $x = $ds->magnitude($vector);

=cut

sub magnitude {
    my ($self, $vector) = @_;
    return sqrt $self->sum_of_squares($vector);
}

=head2 distance

  $x = $ds->distance($vector1, $vector2);

=cut

sub distance {
    my ($self, @vectors) = @_;
    return $self->magnitude($self->vector_subtract(@vectors));
}

=head1 MATRIX METHODS

=head2 shape

  $v = $ds->shape($matrix);

=cut

sub shape {
    my ($self, $matrix) = @_;
    my $rows = @$matrix;
    my $cols = $matrix->[0] ? @{ $matrix->[0] } : 0;
    return [ $rows, $cols ];
}

=head2 get_row

  $v = $ds->get_row($matrix, $i);

=cut

sub get_row {
    my ($self, $matrix, $n) = @_;
    return $matrix->[$n];
}

=head2 get_col

  $v = $ds->get_col($matrix, $i);

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

  $m = $ds->make_matrix($rows, $cols, $entry_fn);

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

L<Data::Science::FromScratch>

F<t/001-linear-algebra.t>

L<List::Util>

L<Moo::Role>

=cut
