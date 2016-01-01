DROP TABLE IF EXISTS classifications;
CREATE TABLE IF NOT EXISTS classifications (
  type_name      TEXT  NOT NULL,
  uid            TEXT  NOT NULL,
  classification BYTEA NOT NULL,

  PRIMARY KEY (uid, type_name)
);
