package Data::Science::FromScratch::NNLinear;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NNLayer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::Linear;

  my $linear = Data::Science::FromScratch::Linear->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::Linear> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 input_dim

  $x = $linear->input_dim;

=cut

has input_dim => (
    is => 'ro',
    required => 1,
);

=head2 output_dim

  $x = $linear->output_dim;

=cut

has output_dim => (
    is => 'ro',
    required => 1,
);

=head2 init

  $x = $linear->init;

Default: C<xavier>

=cut

has init => (
    is      => 'ro',
    lazy    => 1,
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
    lazy    => 1,
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
    lazy    => 1,
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
