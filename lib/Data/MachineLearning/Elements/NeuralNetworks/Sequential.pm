package Data::MachineLearning::Elements::NeuralNetworks::Sequential;

use Moo;
use strictures 2;
use Storable qw(dclone);
use namespace::clean;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements::NeuralNetworks::Sequential;

  my $layer = Data::MachineLearning::Elements::NeuralNetworks::Sequential->new;

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Sequential> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 layers

  $obj = $layer->layers;

=cut

has layers => (
    is       => 'rw',
    required => 1,
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    my $in = dclone $input;
    for my $layer (@{ $self->layers }) {
        $in = $layer->forward($in);
    }
    return $in;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    my $grad = dclone $gradient;
    for my $layer (reverse @{ $self->layers }) {
        $grad = $layer->backward($grad);
    }
    return $grad;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my $self = shift;
    my @params;
    for my $layer (@{ $self->layers }) {
        push @params, @{ $layer->params(@_) };
    }
    return \@params;
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    my @grads;
    for my $layer (reverse @{ $self->layers }) {
        unshift @grads, @{ $layer->grads };
    }
    return \@grads;
}

1;

=head1 SEE ALSO

L<Data::MachineLearning::Elements::NeuralNetworks::Layer>

L<Moo>

=cut
