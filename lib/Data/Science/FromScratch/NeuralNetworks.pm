package Data::Science::FromScratch::NeuralNetworks;

use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  $x = $ds->step_function(); #

=head1 METHODS

=head2 step_function

  $x = $ds->step_function($x); #

=cut

sub step_function {
    my ($self, $x) = @_;
    return $x >= 0 ? 1 : 0;
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/014-neural-networks.t>

L<Moo::Role>

=cut
