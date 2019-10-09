DROP TABLE IF EXISTS descriptor_set;
CREATE TABLE IF NOT EXISTS descriptor_set (
  uid       TEXT  NOT NULL,
  element   BYTEA NOT NULL,

  PRIMARY KEY (uid)
);
