package Data::MachineLearning::Elements::WorkingWithData;

use Moo::Role;
use Storable qw(dclone);
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my ($means, $stddevs) = $ml->scale([-3,-1,1], [-1,0,1], [1,1,1]); # [-1,0,1], [2,1,0]

  my @v = $ml->rescale([-3,-1,1], [-1,0,1], [1,1,1]); # [-1,-1,1], [0,0,1], [1,1,1]

  my $v = $ml->vector_de_mean([1,2], [3,4], [5,6]); # [[-2,-2], [0,0], [2,2]]

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

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

L<t/006-working-with-data.t>

L<Moo::Role>

=cut
