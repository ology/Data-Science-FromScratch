package Data::MachineLearning::Elements::WorkingWithData;

use List::Util qw(sum0);
use Moo::Role;
use Storable qw(dclone);
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my ($means, $stddevs) = $ml->scale([-3,-1,1], [-1,0,1], [1,1,1]); # [-1,0,1], [2,1,0]

  my @v = $ml->rescale([-3,-1,1], [-1,0,1], [1,1,1]); # [-1,-1,1], [0,0,1], [1,1,1]

  my $v = $ml->vector_de_mean([1,2], [3,4], [5,6]); # [[-2,-2], [0,0], [2,2]]

  $v = $ml->direction([1,2,3]); # [0.5774...

  my $y = $ml->directional_variance([[1,2], [3,4]], [1,1]); # 29

  $v = $ml->directional_variance_gradient([[1,2], [3,4]], [1,1]); #

=head1 METHODS

=head2 scale

  ($means, $stddevs) = $ml->scale(@data);

=cut

sub scale {
    my ($self, @data) = @_;
    my $dim = @{ $data[0] };
    my $means = $self->vector_mean(@data);
    my $stdevs = [];
    for my $i (0 .. $dim - 1) {
        push @$stdevs, $self->standard_deviation(map { $_->[$i] } @data);
    }
    return $means, $stdevs;
}

=head2 rescale

  @v = $ml->rescale(@data);

=cut

sub rescale {
    my ($self, @data) = @_;
    my $dim = @{ $data[0] };
    my ($means, $stdevs) = $self->scale(@data);
    my $rescaled = dclone \@data;
    for my $v (@$rescaled) {
        for my $i (0 .. $dim - 1) {
            $v->[$i] = ($v->[$i] - $means->[$i]) / $stdevs->[$i]
                if $stdevs->[$i] > 0;
        }
    }
    return @$rescaled;
}

=head2 vector_de_mean

  $v = $ml->vector_de_mean(@data);

=cut

sub vector_de_mean {
    my ($self, @data) = @_;
    my $mean = $self->vector_mean(@data);
    return [ map { $self->vector_subtract($_, $mean) } @data ];
}

=head2 direction

  $v = $ml->direction($u);

=cut

sub direction {
    my ($self, $u) = @_;
    my $mag = $self->magnitude($u);
    return [ map { $_ / $mag } @$u ];
}

=head2 directional_variance

  $y = $ml->directional_variance($m, $u);

=cut

sub directional_variance {
    my ($self, $m, $u) = @_;
    my $dir = $self->direction($u);
    return sum0(map { $self->vector_dot($_, $dir) ** 2 } @$m);
}

=head2 directional_variance_gradient

  $v = $ml->directional_variance_gradient($m, $u);

=cut

sub directional_variance_gradient {
    my ($self, $m, $u) = @_;
    my $dir = $self->direction($u);
    my @grad;
    for my $i (0 .. @$u - 1) {
        push @grad, sum0(map { 2 * $self->vector_dot($_, $dir) * $_->[$i] } @$m);
    }
    return \@grad;
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

L<t/006-working-with-data.t>

L<List::Util>

L<Moo::Role>

L<Storable>

=cut
