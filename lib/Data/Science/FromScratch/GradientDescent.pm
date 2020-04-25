package Data::Science::FromScratch::GradientDescent;

use Iterator::Simple qw(iterator);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $v = $ds->gradient_step(); # 
  $v = $ds->doubled_gradient(); # 
  $v = $ds->linear_gradient(); # 
  $v = $ds->minibatches(); # 

=head1 METHODS

=head2 gradient_step

  $v = $ds->gradient_step($vector, $gradient, $step_size);

=cut

sub gradient_step {
    my ($self, $vector, $gradient, $step_size) = @_;
    die 'Vectors must be of equal length'
        unless @$vector == @$gradient;
    my $step = $self->scalar_multiply($step_size, $gradient);
    return $self->vector_sum($vector, $step);
}

=head2 doubled_gradient

  $v = $ds->doubled_gradient($vector);

The book calls this method "sum_of_squares_gradient."  But that can't
be correct, since there is no summing and no squaring in the
algorithm!

=cut

sub doubled_gradient {
    my ($self, $vector) = @_;
    my @v;
    push @v, 2 * $_
        for @$vector;
    return \@v;
}

=head2 linear_gradient

  $v = $ds->linear_gradient($vector, $gradient, $step_size);

=cut

sub linear_gradient {
    my ($self, $x, $y, $theta) = @_;
    my ($slope, $intercept) = @$theta;
    my $predicted = $slope * $x + $intercept;
    my $error = $predicted - $y;
    return [2 * $error * $x, 2 * $error];
}

=head2 minibatches

  $v = $ds->minibatches($dataset, $batch_size);

TODO: shuffle

=cut

sub minibatches {
    my ($self, $dataset, $batch_size) = @_;
    my $start = 0;
    iterator {
        my $end = $start + $batch_size - 1;
        $end = @$dataset - 1
            if $end >= @$dataset;
        my @result = @$dataset[$start .. $end];
        return
            unless @result;
        $start += $batch_size;
        return \@result;
    }
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/005-gradient-descent.t>

L<Iterator::Simple>

L<Moo::Role>

=cut
