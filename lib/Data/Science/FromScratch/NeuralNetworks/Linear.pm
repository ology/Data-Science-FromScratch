package Data::Science::FromScratch::NeuralNetworks::Linear;

use List::Util qw(sum0);
use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NeuralNetworks::Layer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::NeuralNetworks::Linear;

  my $layer = Data::Science::FromScratch::NeuralNetworks::Linear->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::Linear> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 input_dim

  $x = $layer->input_dim;

=cut

has input_dim => (
    is => 'ro',
    required => 1,
);

=head2 output_dim

  $x = $layer->output_dim;

=cut

has output_dim => (
    is => 'ro',
    required => 1,
);

=head2 init

  $x = $layer->init;

Default: C<xavier>

=cut

has init => (
    is      => 'ro',
    lazy    => 1,
    default => sub { 'xavier' },
);

=head2 input

  $v = $layer->input;

=cut

has input => (
    is => 'rw',
);

=head2 w

  $v = $layer->w;

Weights

=cut

has w => (
    is      => 'ro',
    lazy    => 1,
    builder => 1,
);

sub _build_w {
    my ($self) = @_;
    return $self->ds->random_tensor([ $self->output_dim, $self->input_dim ], $self->init);
}

=head2 w_grad

  $v = $layer->w_grad;

=cut

has w_grad => (
    is => 'rw',
);

=head2 b

  $v = $layer->b;

Bias

=cut

has b => (
    is      => 'ro',
    lazy    => 1,
    builder => 1,
);

sub _build_b {
    my ($self) = @_;
    return $self->ds->random_tensor([ $self->output_dim ], $self->init);
}

=head2 b_grad

  $v = $layer->b_grad;

=cut

has b_grad => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    $self->input($input);
    return [map { $self->ds->vector_dot($input, $self->w->[$_]) + $self->b->[$_] } 0 .. $self->output_dim - 1];
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    die 'No forward inputs' unless $self->input;
    $self->b_grad($gradient);
    my @w_grad;
    for my $i (0 .. $self->output_dim - 1) {
        push @w_grad, [map { $self->input->[$_] * $gradient->[$i] } 0 .. $self->input_dim - 1];
    }
    $self->w_grad(\@w_grad);
    my @sum;
    for my $i (0 .. $self->input_dim - 1) {
        push @sum, sum0(map { $self->w->[$_][$i] * $gradient->[$_] } 0 .. $self->output_dim - 1);
    }
    return \@sum;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my ($self) = @_;
    return [$self->w, $self->b];
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return [$self->w_grad, $self->b_grad];
}

1;

=head1 SEE ALSO

L<Data::Science::FromScratch::NeuralNetworks::Layer>

=cut
