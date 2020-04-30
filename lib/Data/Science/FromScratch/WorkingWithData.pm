package Data::Science::FromScratch::WorkingWithData;

use Storable qw(dclone);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my ($means, $stddevs) = $ds->scale([-3,-1,1], [-1,0,1], [1,1,1]); # [-1,0,1], [2,1,0]

  my @v = $ds->rescale([-3,-1,1], [-1,0,1], [1,1,1]); # [-1,-1,1], [0,0,1], [1,1,1]

=head1 METHODS

=head2 scale

  ($means, $stddevs) = $ds->scale(@data);

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

  @v = $ds->rescale(@data);

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

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

L<t/006-working-with-data.t>

L<Moo::Role>

=cut
