package Data::Science::FromScratch::DeepLearning;

use Moo::Role;
use Storable qw(dclone);
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $v = $ds->dl_shape(); #

=head1 METHODS

=head2 dl_shape

  $v = $ds->dl_shape($u); #

=cut

sub dl_shape {
    my ($self, $u) = @_;
    my $v = dclone $u;
    my @sizes;
    while (ref $v) {
        push @sizes, scalar @$v;
        $v = $v->[0];
    }
    return \@sizes;
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/015-deep-learning.t>

L<Moo::Role>

=cut
