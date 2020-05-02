package Data::Science::FromScratch::DeepLearning;

use List::Util qw(sum0);
use Moo::Role;
use Storable qw(dclone);
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $v = $ds->tensor_shape([1,2,3]); # [3]

  my $y = $ds->is_1d([1,2,3]); # 1

  $y = $ds->tensor_sum([1,2,3]); # 6

  $v = $ds->tensor_apply(sub { shift() + 1 }, [1,2,3]); # [2,3,4]

  $v = $ds->zeros_like([1,2,3]); # [0,0,0]

  $v = $ds->tensor_combine(sub { shift() * shift() }, [1,2,3], [4,5,6]); # [4,10,18]

=head1 METHODS

=head2 tensor_shape

  $v = $ds->tensor_shape($tensor);

=cut

sub tensor_shape {
    my ($self, $tensor) = @_;
    my $v = dclone $tensor;
    my @sizes;
    while (ref $v) {
        push @sizes, scalar @$v;
        $v = $v->[0];
    }
    return \@sizes;
}

=head2 is_1d

  $y = $ds->is_1d($tensor);

=cut

sub is_1d {
    my ($self, $tensor) = @_;
    return ref $tensor->[0] ? 0 : 1;
}

=head2 tensor_sum

  $y = $ds->tensor_sum($tensor);

=cut

sub tensor_sum {
    my ($self, $tensor) = @_;
    if ($self->is_1d($tensor)) {
        return sum0(@$tensor);
    }
    else {
        return sum0(map { $self->tensor_sum($_) } @$tensor);
    }
}

=head2 tensor_apply

  $v = $ds->tensor_apply($fn, $tensor);

=cut

sub tensor_apply {
    my ($self, $fn, $tensor) = @_;
    if ($self->is_1d($tensor)) {
        return [map { $fn->($_) } @$tensor];
    }
    else {
        return [map { $self->tensor_apply($fn, $_) } @$tensor];
    }
}

=head2 zeros_like

  $v = $ds->zeros_like($tensor);

=cut

sub zeros_like {
    my ($self, $tensor) = @_;
    return $self->tensor_apply(sub { 0 }, $tensor);
}

=head2 tensor_combine

  $v = $ds->tensor_combine($fn, $t1, $t2);

=cut

sub tensor_combine {
    my ($self, $fn, $t1, $t2) = @_;
    if ($self->is_1d($t1)) {
        return [map { $fn->($t1->[$_], $t2->[$_]) } 0 .. @$t1 - 1];
    }
    else {
        return [map { $self->tensor_combine($fn, $t1->[$_], $t2->[$_]) } 0 .. @$t1 - 1];
    }
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/015-deep-learning.t>

L<List::Util>

L<Moo::Role>

L<Storable>

=cut
