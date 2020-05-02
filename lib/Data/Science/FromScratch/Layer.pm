package Data::Science::FromScratch::Layer;

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

  my $x = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
}

=head2 backward

  my $x = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
}

=head2 params

  my $x = $layer->params;

=cut

sub params {
    my ($self) = @_;
    return ();
}

=head2 grads

  my $x = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return ();
}

package Data::Science::FromScratch::Sigmoid;

use Data::Science::FromScratch;
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

1;
