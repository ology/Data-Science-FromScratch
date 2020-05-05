package Data::MachineLearning::Elements::DeepLearning;

use List::Util qw(sum0);
use Moo::Role;
use Storable qw(dclone);
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ds = Data::MachineLearning::Elements->new;

  my $v = $ds->tensor_shape([1,2,3]); # [3]

  my $y = $ds->is_1d([1,2,3]); # 1

  $y = $ds->tensor_sum([1,2,3]); # 6

  $v = $ds->tensor_apply(sub { shift() + 1 }, [1,2,3]); # [2,3,4]

  $v = $ds->zeros_like([1,2,3]); # [0,0,0]

  $v = $ds->tensor_combine(sub { shift() * shift() }, [1,2,3], [4,5,6]); # [4,10,18]

  $v = $ds->random_uniform(); #
  $v = $ds->random_normal(); #

  $v = $ds->random_tensor(); #

  $y = $ds->tanh(0); # 0
  $y = $ds->tanh(1); # 0.7616

=head1 DESCRIPTION

There is something amiss when running a NN with this module.  Comment-out the
C<skip> blocks in L<the test|/"SEE ALSO"> to see the error that is generated.

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

=head2 random_uniform

  $v = $ds->random_uniform($dims);

=cut

sub random_uniform {
    my ($self, $dims) = @_;
    if (@$dims == 1) {
        return [map { rand } 0 .. $dims->[0] - 1];
    }
    else {
        return [map { $self->random_uniform([ @$dims[1 .. @$dims - 1] ]) } 0 .. $dims->[0] - 1];
    }
}

=head2 random_normal

  $v = $ds->random_normal($dims, $mean, $variance);

=cut

sub random_normal {
    my ($self, $dims, $mean, $variance) = @_;
    $mean ||= 0;
    $variance ||= 1;
    if (@$dims == 1) {
        return [map { $mean + $variance * $self->inverse_normal_cdf(rand) } 0 .. $dims->[0] - 1];
    }
    else {
        return [map { $self->random_normal([ @$dims[1 .. @$dims - 1] ], $mean, $variance) } 0 .. $dims->[0] - 1];
    }
}

=head2 random_tensor

  $v = $ds->random_tensor($dims, $init);

=cut

sub random_tensor {
    my ($self, $dims, $init) = @_;
    if ($init eq 'normal') {
        return $self->random_normal($dims);
    }
    elsif ($init eq 'uniform') {
        return $self->random_uniform($dims);
    }
    elsif ($init eq 'xavier') {
        my $variance = @$dims / sum0(@$dims);
        return $self->random_normal($dims, 0, $variance);
    }
    else {
        die "Unknown init: $init";
    }
}

=head2 tanh

  $y = $ds->tanh($x);

=cut

sub tanh {
    my ($self, $x) = @_;
    if ($x < -100) {
        return -1;
    }
    elsif ($x > 100) {
        return 1;
    }
    my $em2x = exp(-2 * $x);
    return (1 - $em2x) / (1 + $em2x);
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/015-deep-learning.t>

L<List::Util>

L<Moo::Role>

L<Storable>

=cut
