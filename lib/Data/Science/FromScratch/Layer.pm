package Data::Science::FromScratch::Layer;

use Data::Science::FromScratch;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  use Data::Science::FromScratch::Layer;

  my $layer = Data::Science::FromScratch::Layer->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::Layer> is an abstract class for building neural networks.

=head1 ATTRIBUTES

=head2 ds

  $obj = $layer->ds;

C<Data::Science::FromScratch> object

=cut

has ds => (
    is        => 'ro',
    init_args => undef,
    default   => sub { Data::Science::FromScratch->new },
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my ($self) = @_;
    return ();
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return ();
}

package Data::Science::FromScratch::Sigmoid;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::Layer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::Sigmoid;

  my $sigmoid = Data::Science::FromScratch::Sigmoid->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::Sigmoid> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 sigmoids

  $v = $sigmoid->sigmoids;

=cut

has sigmoids => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $sigmoid->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    my $sigmoids = $self->ds->tensor_apply(sub { return $self->ds->sigmoid(shift()) }, $input);
    $self->sigmoids($sigmoids);
    return $self->sigmoids;
}

=head2 backward

  $v = $sigmoid->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    return $self->ds->tensor_combine(
        sub { my ($sig, $grad) = @_; return $sig * (1 - $sig) * $grad },
        $self->sigmoids,
        $gradient
    );
}

package Data::Science::FromScratch::Linear;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::Layer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::Linear;

  my $linear = Data::Science::FromScratch::Linear->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::Linear> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 input_dim

  $v = $linear->input_dim;

=cut

has input_dim => (
    is => 'ro',
);

=head2 output_dim

  $v = $linear->output_dim;

=cut

has output_dim => (
    is => 'ro',
);

=head2 init

  $x = $linear->init;

Default: C<xavier>

=cut

has init => (
    is      => 'ro',
    default => sub { 'xavier' },
);

=head2 input

  $v = $linear->input;

=cut

has input => (
    is => 'rw',
);

=head2 w

  $v = $linear->w;

Weights

=cut

has w => (
    is      => 'ro',
    builder => 1,
);

sub _build_w {
    my ($self) = @_;
    return $self->ds->random_tensor([ $self->output_dim, $self->input_dim ], $self->init);
}

=head2 w_grad

  $v = $linear->w_grad;

=cut

has w_grad => (
    is => 'rw',
);

=head2 b

  $v = $linear->b;

Bias

=cut

has b => (
    is      => 'ro',
    builder => 1,
);

sub _build_b {
    my ($self) = @_;
    return $self->ds->random_tensor([ $self->output_dim ], $self->init);
}

=head2 b_grad

  $v = $linear->b_grad;

=cut

has b_grad => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $linear->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    $self->input($input);
    return [map { $self->ds->vector_dot($input, $self->w->[$_]) } 0 .. $self->output_dim];
}

=head2 backward

  $v = $linear->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    $self->b_grad($gradient);
    my @w_grad;
    for my $i (0 .. $self->output_dim) {
        push @w_grad, [map { $self->input->[$_] * $gradient->[$i] } 0 .. $self->input_dim];
    }
    $self->w_grad(\@w_grad);
    my @sum;
    for my $i (0 .. $self->input_dim) {
        push @sum, [map { $self->w->[$_][$i] * $gradient->[$_] } 0 .. $self->output_dim];
    }
    return \@sum;
}

=head2 params

  $v = $linear->params;

=cut

sub params {
    my ($self) = @_;
    return [$self->w, $self->b];
}

=head2 grads

  $v = $linear->grads;

=cut

sub grads {
    my ($self) = @_;
    return [$self->w_grad, $self->b_grad];
}

1;
